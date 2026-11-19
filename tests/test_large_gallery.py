import gzip
import json
from hashlib import sha256

import numpy as np
import pytest
from PIL import Image

from cardlens_ai import large_gallery as gallery_module


@pytest.fixture
def gallery_bundle(tmp_path, monkeypatch):
    """Small pinned bundle; real .npy/gzip parsing and FAISS, stubbed tower only."""
    (tmp_path / "index").mkdir()
    (tmp_path / "models").mkdir()
    dims = gallery_module.EMBED_DIM
    duplicates = [f"alternate-{i}" for i in range(10)]
    handles = ["variant", *duplicates, "view", "other-print", "ambiguous-view", "not-in-catalog", "wrong-print", "wrong-view"]
    vectors = np.zeros((len(handles), dims), dtype=np.float16)
    vectors[:, 0] = 3
    vectors[1:11, 1] = 1
    vectors[11, :2] = [3, 4]
    vectors[12, :2] = [3, 4]
    np.save(tmp_path / "index/embeddings.npy", vectors)

    set_a = "international/example"
    set_b = "zh-hans/example"
    print_a = {"set": set_a, "card": "first", "number": "4/102"}
    print_b = {"set": set_b, "card": "second", "number": "No. 001", "language": "zh-hans"}
    # JSONL order is not the matrix row order.
    cards = [
        {"handle": "wrong-print", **print_b, "language": "zh-Hant"},
        {"handle": "other-print", **print_b},
        {"handle": "ambiguous-view", "kind": "view"},
        {"handle": "not-in-catalog", "set": set_a, "card": "third"},
        {"handle": "view", "kind": "view"},
        {"handle": "wrong-view", "kind": "view", "language": "fr"},
        {"handle": "variant", "variant": "holo", **print_a},
        *[{"handle": handle, **print_a} for handle in duplicates],
    ]
    with gzip.open(tmp_path / "index/cards.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.writelines(json.dumps(card) + "\n" for card in cards)
    groups = {handle: "base" for handle in handles}
    for handle in ("other-print", "wrong-print", "ambiguous-view", "not-in-catalog"):
        groups[handle] = "ambiguous"
    snapshot_sha = "1" * 64
    mapping = {"schema_version": 1, "snapshot_sha256": snapshot_sha, "catalog_variants": 3, "printings": {
        f"{set_a}/first": {"card_id": "legacy:004:en", "set_id": "legacy", "name": "One", "set": "Example",
                           "language": "en", "number": "004", "official_total": None,
                           "variant_options": [{"variantId": "holo"}, {"variantId": "normal"}]},
        f"{set_b}/second": {"card_id": "local:second:zh-Hans", "set_id": "local", "name": "Two", "set": "Example",
                            "language": "zh-Hans", "number": None, "number_is_printed": False,
                            "official_total": "102", "variant_options": ["normal"]},
    }}
    for name, value in {
        "index/rows.json": handles, "index/dupe-groups.json": groups, "mapping.json": mapping,
        "index/release.json": {"scanner": gallery_module.RELEASE, "pictures": len(handles)},
        "pipeline.json": {"version": gallery_module.RELEASE, "embedder": gallery_module.EMBEDDER_SPEC},
    }.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    for name in ("models/matcher.safetensors", "models/heads.safetensors"):
        (tmp_path / name).write_bytes(b"fixture, not loaded")

    def checksum(name):
        return sha256((tmp_path / name).read_bytes()).hexdigest()

    upstream = {name: checksum(name) for name in gallery_module.UPSTREAM_SHA256}
    manifest = {**upstream, "mapping.json": checksum("mapping.json")}
    (tmp_path / "checksums.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(gallery_module, "UPSTREAM_SHA256", upstream)
    monkeypatch.setattr(gallery_module, "REFERENCE_IMAGES", len(handles))
    monkeypatch.setattr(gallery_module, "_load_embedder", lambda _path, _device: lambda _image: vectors[:1])
    return tmp_path, manifest, vectors


def test_search_returns_mapped_printings_and_excludes_unmatched(gallery_bundle):
    root, _, vectors = gallery_bundle
    gallery = gallery_module.load_gallery(root)
    assert gallery.coverage == {
        "reference_images": len(vectors), "mapped_reference_images": len(vectors) - 4,
        "indexed_printings": 2, "indexed_catalog_variants": 3, "catalog_variants": 3,
        "snapshot_sha256": "1" * 64,
    }
    assert gallery.index.ntotal == len(vectors) - 4
    result = gallery.search(Image.new("RGB", (224, 224)), top_k=2)
    assert set(result) == {"candidates", "timing_ms", "model_versions"}
    best, other = result["candidates"]
    assert best["card_id"] == "legacy:004:en"
    assert best["number"] == "004" and best["official_total"] is None
    assert best["number_is_printed"] is True
    assert best["variant"] is None and best["reference_variant"] == "holo"
    assert best["similarity"] == pytest.approx(1)
    assert other["card_id"] == "local:second:zh-Hans"
    assert other["language"] == "zh-Hans" and other["number"] is None
    assert other["official_total"] == "102" and other["number_is_printed"] is False
    assert gallery.rows[11]["card_id"] == "legacy:004:en" and gallery.rows[11]["reference_variant"] is None
    assert gallery.rows[12]["card_id"] == "local:second:zh-Hans"
    assert gallery.rows[13:] == [None, None, None, None]
    with pytest.raises(ValueError, match="top_k"):
        gallery.search(Image.new("RGB", (224, 224)), top_k=51)


def test_cached_gallery_reloads_after_verified_mapping_changes(gallery_bundle):
    root, manifest, _ = gallery_bundle
    first = gallery_module.get_gallery(root)
    mapping = json.loads((root / "mapping.json").read_text(encoding="utf-8"))
    printing = mapping["printings"]["international/example/first"]
    printing["source_variant_options"] = ["etched"]
    mapping["catalog_variants"] += 1
    (root / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    manifest["mapping.json"] = sha256((root / "mapping.json").read_bytes()).hexdigest()
    (root / "checksums.json").write_text(json.dumps(manifest), encoding="utf-8")

    second = gallery_module.get_gallery(root)

    assert second is not first
    assert second.coverage["catalog_variants"] == first.coverage["catalog_variants"] + 1
    assert second.coverage["indexed_catalog_variants"] == first.coverage["indexed_catalog_variants"] + 1

def test_manifest_does_not_allow_repinning_upstream_artifacts(gallery_bundle):
    root, manifest, _ = gallery_bundle
    name = "models/matcher.safetensors"
    (root / name).write_bytes(b"different matcher")
    manifest[name] = sha256(b"different matcher").hexdigest()
    (root / "checksums.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum_mismatch.*matcher"):
        gallery_module.load_gallery(root)


def test_mapping_requires_verified_hash(gallery_bundle):
    root, manifest, _ = gallery_bundle
    (root / "mapping.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum_mismatch.*mapping"):
        gallery_module.load_gallery(root)
    manifest.pop("mapping.json")
    (root / "checksums.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksums"):
        gallery_module.load_gallery(root)


@pytest.mark.parametrize("failure", ["rows", "dimension", "nan", "zero", "pickle"])
def test_invalid_index_vectors_are_rejected_before_model_load(gallery_bundle, monkeypatch, failure):
    root, manifest, vectors = gallery_bundle
    invalid = {"rows": vectors[:-1], "dimension": vectors[:, :-1], "pickle": vectors.astype(object)}
    if failure == "nan":
        vectors[-1, 0] = np.nan
    if failure == "zero":
        vectors[-1] = 0
    name = "index/embeddings.npy"
    path = root / name
    np.save(path, invalid.get(failure, vectors))
    digest = sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(gallery_module.UPSTREAM_SHA256, name, digest)
    (root / "checksums.json").write_text(json.dumps({**manifest, name: digest}), encoding="utf-8")
    monkeypatch.setattr(gallery_module, "_load_embedder", lambda *_: pytest.fail("invalid index reached model loading"))
    with pytest.raises(ValueError):
        gallery_module.load_gallery(root)


def test_printless_view_language_must_match_its_single_group_print():
    mapping = {"schema_version": 1, "snapshot_sha256": "1" * 64, "catalog_variants": 1,
               "printings": {"international/set/1": {"card_id": "local:1:en", "set_id": "set",
                   "language": "en", "number": None, "number_is_printed": False,
                   "official_total": "99", "name": "One", "set": "Set", "variant_options": ["normal"]}}}
    cards = {"one": {"set": "international/set", "card": "1", "language": "en"},
             "good-view": {"kind": "view", "language": "EN"},
             "wrong-view": {"kind": "view", "language": "fr"}}
    rows = gallery_module._mapped_rows(list(cards), cards, {"one": "single", "good-view": "single", "wrong-view": "single"}, mapping)
    assert rows[0]["number"] is None and rows[0]["number_is_printed"] is False
    assert rows[0]["official_total"] == "99"
    assert rows[1]["card_id"] == "local:1:en"
    assert rows[2] is None

