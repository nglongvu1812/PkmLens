"""Offline CardCorpus matcher paired with its pinned reference-image index.

Architecture and preprocessing: CardCorpus's Apache-2.0 scanner, tower.py at
2a2db34fe0615b5f757e7a75621a6cd9cbf49d07. Built with DINOv3.
"""
import gzip
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock

import faiss
import numpy as np

from .identity import contained_path, json_fingerprint, sha256

MODEL_REVISION = "6ff14bfa83dc29fd444fcd7c043622f0d6fcbe49"
SCANNER_REVISION = "2a2db34fe0615b5f757e7a75621a6cd9cbf49d07"
RELEASE = "2026-11-11"
REFERENCE_IMAGES = 102_185
EMBED_DIM = 512
EMBEDDER_SPEC = {
    "arch": "vit_small_patch16_dinov3.lvd1689m", "size": 224,
    "interpolation": "bilinear", "resize_mode": "shortest",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "head": None, "projection": 384, "dim": EMBED_DIM, "logit_limit": 64000.0,
}
# Upstream SHA256SUMS at MODEL_REVISION. The local manifest cannot re-pin these.
UPSTREAM_SHA256 = {
    "pipeline.json": "2fdd4da588ecf57b1ed49578f64ce1e23d2db1b80d26bde11b039145370d7de3",
    "index/release.json": "8b0cf01eda2d31f04d0f2e01b1a1d785002a61d76d9c9c35cf0bba3ac0aa614d",
    "index/embeddings.npy": "94e597c177afe4b8f3579835a3eb108c9560637cbab6a54c6c175cb03c47c3c1",
    "index/rows.json": "af0be2f5f8a2837100486b844f0f2d385c259da7a7093fc3f8c107fad82fc779",
    "index/cards.jsonl.gz": "373045fd093f32d5f7643836edd39f1487efec305a79958530b8fbd1852da466",
    "index/dupe-groups.json": "c44414117a3f4ffdbfa64dd8189cd09ab9966760a02e24db14e2037b62cfb539",
    "models/matcher.safetensors": "f7a8f7ae39647f4174e850ecf0ea4cf777c6f4987103ef26002749dc4249b721",
    "models/heads.safetensors": "eb4d01472685e71d4f84bc33dab19a2dad15d1c7efc04502d9b905f68faec877",
}
_load_lock = Lock()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _verify_manifest(root):
    manifest = _json(contained_path(root, "checksums.json"))
    required = set(UPSTREAM_SHA256) | {"mapping.json"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("large_gallery_missing_or_unexpected_checksums")
    for name, expected in manifest.items():
        if (not _is_sha256(expected) or expected != UPSTREAM_SHA256.get(name, expected)
                or sha256(contained_path(root, name)) != expected):
            raise ValueError(f"large_gallery_checksum_mismatch: {name}")
    return manifest


def _text(value):
    return value if isinstance(value, str) and value.strip() else None


def _variant_id(option):
    if isinstance(option, dict):
        return _text(option.get("variantId")) or _text(option.get("id"))
    return _text(option)


def _mapped_rows(handles, cards, groups, mapping):
    if (not isinstance(mapping, dict) or type(mapping.get("schema_version")) is not int
            or mapping["schema_version"] != 1 or not isinstance(mapping.get("printings"), dict)
            or not _is_sha256(mapping.get("snapshot_sha256"))
            or type(mapping.get("catalog_variants")) is not int or mapping["catalog_variants"] < 0):
        raise ValueError("large_gallery_invalid_mapping")
    printings = mapping["printings"]
    seen_ids = set()
    for printing, entry in printings.items():
        if (not _text(printing) or not isinstance(entry, dict)
                or not all(_text(entry.get(key)) for key in ("card_id", "set_id", "language", "name", "set"))
                or entry["card_id"] in seen_ids
                or any(entry.get(key) is not None and not _text(entry[key]) for key in ("number", "official_total"))
                or ("number_is_printed" in entry and (type(entry["number_is_printed"]) is not bool
                    or entry["number_is_printed"] != (entry.get("number") is not None)))
                or not isinstance(entry.get("variant_options"), list)
                or not isinstance(entry.get("source_variant_options", []), list)
                or not isinstance(entry.get("set_codes", []), list)
                or any(not _text(code) for code in entry.get("set_codes", []))
                or any(not _variant_id(option) for option in entry["variant_options"])
                or any(not _variant_id(option) for option in entry.get("source_variant_options", []))):
            raise ValueError("large_gallery_invalid_printing_mapping")
        seen_ids.add(entry["card_id"])

    owned = {}
    by_group = {}
    for handle, card in cards.items():
        printing = _text(card.get("print"))
        if not printing and _text(card.get("set")) and _text(card.get("card")):
            printing = f"{card['set']}/{card['card']}"
        if printing:
            owned[handle] = printing
            if group := groups.get(handle):
                by_group.setdefault(group, {}).setdefault(printing, card)

    rows = []
    for handle in handles:
        reference = card = cards[handle]
        printing = owned.get(handle)
        if not printing:
            siblings = by_group.get(groups.get(handle), {})
            # Only a single source printing with compatible language can lend identity to a view.
            if len(siblings) == 1:
                printing, card = next(iter(siblings.items()))
        entry = printings.get(printing)
        ref_language = _text(reference.get("language"))
        if not entry or (ref_language and ref_language.casefold() != entry["language"].casefold()):
            rows.append(None)
            continue
        rows.append({
            **{key: entry[key] for key in ("card_id", "name", "set", "set_id", "language", "variant_options")},
            "source_variant_options": entry.get("source_variant_options", []),
            "number": entry.get("number"),
            "number_is_printed": entry.get("number_is_printed", entry.get("number") is not None),
            "official_total": entry.get("official_total"), "set_codes": entry.get("set_codes") or [],
            "variant": None, "reference_variant": _text(reference.get("variant")),
        })
    return rows


def _read_rows(files):
    handles = _json(files["index/rows.json"])
    groups = _json(files["index/dupe-groups.json"])
    if (not isinstance(handles, list) or len(handles) != REFERENCE_IMAGES
            or any(not _text(handle) for handle in handles) or len(set(handles)) != len(handles)
            or not isinstance(groups, dict) or set(groups) != set(handles)
            or any(not _text(group) for group in groups.values())):
        raise ValueError("large_gallery_row_alignment_mismatch")
    cards = {}
    with gzip.open(files["index/cards.jsonl.gz"], "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            handle = row.get("handle") if isinstance(row, dict) else None
            if not _text(handle) or handle in cards:
                raise ValueError("large_gallery_invalid_card_rows")
            cards[handle] = row
    if not set(handles) <= cards.keys():
        raise ValueError("large_gallery_row_alignment_mismatch")
    return handles, cards, groups


def _normalise_embeddings(values):
    if values.ndim != 2 or values.shape[1] != EMBED_DIM:
        raise ValueError("large_gallery_embedding_dimension_mismatch")
    vectors = np.array(values, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(vectors).all():
        raise ValueError("large_gallery_nonfinite_embeddings")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("large_gallery_invalid_embedding_norm")
    vectors /= norms[:, None]
    return vectors


def _index(path, rows):
    values = np.load(path, allow_pickle=False, mmap_mode="r")
    if not isinstance(values, np.ndarray) or values.shape != (len(rows), EMBED_DIM) or values.dtype != np.float16:
        raise ValueError("large_gallery_embedding_row_alignment_mismatch")
    index, positions = faiss.IndexFlatIP(EMBED_DIM), []
    for start in range(0, len(rows), 8192):
        chunk = _normalise_embeddings(values[start:start + 8192])
        keep = [offset for offset, row in enumerate(rows[start:start + 8192]) if row is not None]
        if keep:
            index.add(np.ascontiguousarray(chunk[keep]))
            positions.extend(start + offset for offset in keep)
    return index, positions


def _load_embedder(path, device):
    import importlib.metadata
    import sys

    local_dependencies = Path(path).resolve().parents[3] / "verification-deps" / "cardcorpus"
    local_timm = local_dependencies / "timm" / "__init__.py"
    if local_timm.is_file() and str(local_dependencies) not in sys.path:
        sys.path.insert(0, str(local_dependencies))
    if importlib.metadata.version("timm") != "1.0.30":
        raise RuntimeError("cardcorpus_timm_1_0_30_required")

    import timm
    import torch
    from safetensors.torch import load_file
    from torch import nn
    from torchvision import transforms as T

    device = torch.device(device)
    spec, size = EMBEDDER_SPEC, EMBEDDER_SPEC["size"]
    tower = nn.ModuleDict({
        "backbone": timm.create_model(spec["arch"], pretrained=False, num_classes=0, img_size=(size, size)),
        "proj": nn.Linear(spec["projection"], spec["dim"], bias=False),
    })
    tower.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    tower.float().eval().requires_grad_(False).to(device)
    preprocess = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop((size, size)), T.ToTensor(), T.Normalize(spec["mean"], spec["std"]),
    ])

    from .detection import model_lock, run_on_gpu

    def _forward(image):
        # FP32 avoids the upstream 16-bit overflow workaround that replaces global SDPA.
        with model_lock, torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            batch = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
            return tower["proj"](tower["backbone"](batch)).cpu().numpy()

    def embed(image):
        return run_on_gpu(_forward, image) if device.type != "cpu" else _forward(image)

    return embed


class LargeGallery:
    def __init__(self, index, rows, index_rows, embed, model_versions, coverage):
        self.index, self.rows, self.index_rows = index, rows, index_rows
        self.embed, self.model_versions, self.coverage = embed, model_versions, coverage

    def search(self, image, top_k=50):
        """Search one upright, already rectified crop. Similarities are not identities."""
        if type(top_k) is not int or not 1 <= top_k <= 50:
            raise ValueError("invalid_large_gallery_top_k")
        started = time.perf_counter()
        query = _normalise_embeddings(self.embed(image))
        if len(query) != 1:
            raise ValueError("large_gallery_invalid_query_count")
        embedded = time.perf_counter()
        wanted = min(top_k, self.coverage["indexed_printings"])
        count = min(wanted * 4, self.index.ntotal)
        while True:
            scores, indices = self.index.search(query, count)
            candidates = {}
            for score, i in zip(scores[0], indices[0]):
                row = self.rows[self.index_rows[i]] if i >= 0 else None
                if row and row["card_id"] not in candidates:
                    candidates[row["card_id"]] = {**row, "similarity": float(score)}
                    if len(candidates) == wanted:
                        break
            if len(candidates) == wanted or count == self.index.ntotal:
                break
            count = min(count * 2, self.index.ntotal)
        return {
            "candidates": list(candidates.values()),
            "timing_ms": {"embedding": (embedded - started) * 1000,
                          "retrieval": (time.perf_counter() - embedded) * 1000},
            "model_versions": dict(self.model_versions),
        }


def load_gallery(root, *, device="cpu") -> LargeGallery:
    """Validate local artifacts and build the paired matcher and exact cosine index."""
    root = Path(root).resolve()
    hashes = _verify_manifest(root)
    files = {name: contained_path(root, name) for name in set(UPSTREAM_SHA256) | {"mapping.json"}}
    pipeline, release = _json(files["pipeline.json"]), _json(files["index/release.json"])
    if (pipeline.get("version") != RELEASE or pipeline.get("embedder") != EMBEDDER_SPEC
            or release.get("scanner") != RELEASE or release.get("pictures") != REFERENCE_IMAGES):
        raise ValueError("large_gallery_incompatible_release")
    mapping = _json(files["mapping.json"])
    handles, cards, groups = _read_rows(files)
    rows = _mapped_rows(handles, cards, groups, mapping)
    del handles, cards, groups
    printings = {row["card_id"] for row in rows if row}
    indexed_variants = {(row["card_id"], _variant_id(option)) for row in rows if row
                        for option in (*row["variant_options"], *row["source_variant_options"])}
    coverage = {
        "reference_images": len(rows), "mapped_reference_images": sum(row is not None for row in rows),
        "indexed_printings": len(printings), "indexed_catalog_variants": len(indexed_variants),
        "catalog_variants": mapping["catalog_variants"], "snapshot_sha256": mapping["snapshot_sha256"],
    }
    if not coverage["indexed_printings"]:
        raise ValueError("large_gallery_no_mapped_printings")
    del mapping, indexed_variants, printings
    versions = {
        "embedding": "cardcorpus/dinov3-small-matcher", "kind": "external_pretrained_matcher",
        "gallery": json_fingerprint({name: hashes[name] for name in files if name.startswith("index/") or name == "mapping.json"}),
        "identity_fingerprint": json_fingerprint({"artifacts": {name: hashes[name] for name in files},
                                                  "preprocess": "shortest-center-crop224-bilinear-imagenet-fp32-v1"}),
        "matcher_revision": MODEL_REVISION, "scanner_code_revision": SCANNER_REVISION,
        "index_release": RELEASE, "device": str(device),
    }
    index, index_rows = _index(files["index/embeddings.npy"], rows)
    embed = _load_embedder(files["models/matcher.safetensors"], device)
    return LargeGallery(index, rows, index_rows, embed, versions, coverage)


@lru_cache(maxsize=2)
def _cached_gallery(root, device, manifest_hash, mapping_size, mapping_mtime):
    return load_gallery(root, device=device)


def get_gallery(root, *, device="cpu") -> LargeGallery:
    """Reuse the verified gallery until its manifest or paired mapping changes."""
    root = Path(root).resolve()
    manifest = contained_path(root, "checksums.json")
    mapping = contained_path(root, "mapping.json")
    stat = mapping.stat()
    signature = (sha256(manifest), stat.st_size, stat.st_mtime_ns)
    with _load_lock:
        return _cached_gallery(root, device, *signature)
