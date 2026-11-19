import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.pose_dataset import prepare_scenes, validate_pose_dataset
from ml.prepare_pose import prepare, validate_pose_dataset as public_validate


SPLITS = ("train", "val", "test")
QUAD = [[20, 10], [180, 10], [180, 90], [20, 90]]


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def scenes(tmp_path, *, test=True, negative=True):
    source = tmp_path / "source"
    rows = []
    for index, split in enumerate(SPLITS if test else SPLITS[:2]):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
        objects = [{"group": "art-" + split, "corners": QUAD, "source": split + ".webp"},
                   {"group": "other-" + split, "corners": [[140, 80], [60, 80], [60, 20], [140, 20]]}]
        for number in range(2 if negative else 1):
            sample = f"{split}-{number}"
            image = source / "images" / split / (sample + ".jpg")
            Image.new("RGB", (200, 100), (20 + index * 70, 30 + number * 90, 80)).save(image)
            label = source / "labels" / split / (sample + ".txt")
            label.write_text("0 .5 .5 .8 .8\n0 .5 .5 .4 .6" if number == 0 else "", encoding="utf-8")
            rows.append({"sample_id": sample, "split": split, "source_kind": "synthetic_catalog_scene",
                         "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                         "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
                         "objects": objects if number == 0 else []})
    write_rows(source / "manifest.jsonl", rows)
    return source, rows


def exported(tmp_path, **kwargs):
    source, rows = scenes(tmp_path, **kwargs)
    output = tmp_path / "pose"
    summary = prepare_scenes(source, output)
    return source, rows, output, summary


def test_shared_validator_export_preserves_bytes_metadata_and_semantic_corner_order(tmp_path):
    source, rows, output, summary = exported(tmp_path)
    assert public_validate is validate_pose_dataset
    assert summary["counts"] == {split: 2 for split in SPLITS}
    result = public_validate(output)
    assert result["valid"], result
    assert result["counts"] == {split: {"images": 2, "objects": 2} for split in SPLITS}
    assert result["source_kinds"] == ["synthetic_catalog_scene"]
    assert len(result["content_sha256"]) == 64
    assert result["split_roles"]["val"] != result["split_roles"]["test"]
    for row in rows:
        name = hashlib.sha256(row["sample_id"].encode()).hexdigest()
        assert (output / "images" / row["split"] / (name + ".jpg")).read_bytes() == (source / "images" / row["split"] / (row["sample_id"] + ".jpg")).read_bytes()
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert records[0]["objects"] == rows[0]["objects"]
    assert records[0]["bbox_label_sha256"] == rows[0]["label_sha256"]


def test_no_test_split_is_supported_with_explicit_warning(tmp_path):
    _, _, output, summary = exported(tmp_path, test=False)
    result = public_validate(output)
    assert result["valid"], result
    assert "test" not in result["counts"]
    assert any("test" in warning and "validation" in warning for warning in result["warnings"])
    assert summary["counts"]["test"] == 0


@pytest.mark.parametrize("change", ["download", "root", "split_file", "split_alias", "flip", "extra_key"])
def test_reject_yaml_that_ultralytics_would_read_differently(tmp_path, change):
    _, _, output, _ = exported(tmp_path)
    config_path = output / "pose.yaml"
    config = json.loads(config_path.read_text())
    if change == "download": config["download"] = "print('must never run')"
    elif change == "root": config["path"] = str(tmp_path)
    elif change == "split_file":
        (output / "train.txt").write_text("outside.jpg")
        config["train"] = "train.txt"
    elif change == "split_alias": config["val"] = "images/train"
    elif change == "flip": config["flip_idx"] = [0, 1, 2, 3]
    else: config["minival"] = "../outside"
    config_path.write_text(json.dumps(config))
    assert not public_validate(output)["valid"]


@pytest.mark.parametrize("change", ["orphan_label", "extra_image", "missing_image", "duplicate_row", "npy", "cache", "nested", "masks"])
def test_reject_unverified_files_rows_and_loader_caches(tmp_path, change):
    _, _, output, _ = exported(tmp_path)
    if change == "orphan_label": (output / "labels/train/orphan.txt").write_text("")
    elif change == "extra_image": Image.new("RGB", (100, 100)).save(output / "images/train/extra.bmp")
    elif change == "missing_image": next((output / "images/train").glob("*.jpg")).unlink()
    elif change == "duplicate_row":
        path = output / "manifest.jsonl"
        path.write_text(path.read_text() + "\n" + path.read_text().splitlines()[0])
    elif change == "npy": np.save(next((output / "images/train").glob("*.jpg")).with_suffix(".npy"), np.zeros((100, 200, 3)))
    elif change == "cache": (output / "labels/train.cache").write_bytes(b"stale loader cache")
    elif change == "nested":
        nested = output / "images/train/nested"
        nested.mkdir()
        Image.new("RGB", (100, 100)).save(nested / "unverified.jpg")
    else: (output / "masks").mkdir()
    assert not public_validate(output)["valid"]


@pytest.mark.parametrize("change", ["zero_box", "wrong_box", "xy_swap", "reverse", "concave", "visibility", "extra_line", "nan"])
def test_reject_bad_labels_and_manifest_geometry(tmp_path, change):
    _, _, output, _ = exported(tmp_path, negative=False)
    label = next((output / "labels/train").glob("*.txt"))
    lines = label.read_text().splitlines()
    values = list(map(float, lines[0].split()))
    if change == "zero_box": values[3] = 0
    elif change == "wrong_box": values[3] = .4
    elif change == "xy_swap": values[5:7] = values[6:4:-1]
    elif change == "visibility": values[7] = 1
    elif change == "nan": values[1] = float("nan")
    elif change == "extra_line": lines.append(lines[0])
    else:
        path = output / "manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["objects"][0]["corners"] = QUAD[::-1] if change == "reverse" else [[20, 10], [180, 10], [60, 30], [20, 90]]
        write_rows(path, rows)
    lines[0] = " ".join(map(str, values))
    label.write_text("\n".join(lines))
    result = public_validate(output)
    assert not result["valid"]
    assert any("pose_label" in error or "corners" in error for error in result["errors"])


@pytest.mark.parametrize("field", ["group", "artwork_id", "session_id", "physical_card_id", "sha256"])
def test_cross_split_leakage_at_image_and_object_level(tmp_path, field):
    source, rows = scenes(tmp_path, negative=False)
    if field == "sha256":
        image = source / "images/val/val-0.jpg"
        image.write_bytes((source / "images/train/train-0.jpg").read_bytes())
        rows[1]["sha256"] = rows[0]["sha256"]
    else:
        for row in rows[:2]: row["objects"][0][field] = "shared"
    write_rows(source / "manifest.jsonl", rows)
    with pytest.raises(ValueError, match="leakage|duplicate_image"):
        prepare_scenes(source, tmp_path / "pose")


@pytest.mark.parametrize("change", ["bad_hash", "bad_label_hash", "bad_box", "duplicate_id", "traversal", "real_without_consent"])
def test_scene_source_validation_happens_before_export(tmp_path, change):
    source, rows = scenes(tmp_path)
    if change == "bad_hash": rows[0]["sha256"] = "0" * 64
    elif change == "bad_label_hash": rows[0]["label_sha256"] = "0" * 64
    elif change == "bad_box":
        path = source / "labels/train/train-0.txt"
        path.write_text("0 .5 .5 .1 .1\n0 .5 .5 .4 .6")
        rows[0].pop("label_sha256")
    elif change == "duplicate_id": rows[1]["sample_id"] = rows[0]["sample_id"]
    elif change == "traversal": rows[0]["sample_id"] = "../../outside"
    else: rows[0]["source_kind"] = "real_photo"
    write_rows(source / "manifest.jsonl", rows)
    output = tmp_path / "pose"
    with pytest.raises(ValueError): prepare_scenes(source, output)
    assert not output.exists()


def test_photo_export_preserves_source_records_consent_and_records_checksums(tmp_path):
    rows = []
    for index, split in enumerate(SPLITS[:2]):
        image = tmp_path / (split + ".jpg")
        Image.new("RGB", (200, 100), (index * 160, 20, 60)).save(image)
        rows.append({"sample_id": split, "session_id": split, "physical_card_id": split,
                     "card_id": "set:" + split, "language": "en", "side": "front", "variant": "unknown",
                     "path": str(image), "source": "owner-photo", "source_kind": "real_photo",
                     "consent": "research_and_evaluation", "corners": QUAD, "split": split,
                     "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "review": {"status": "reviewed"}})
    manifest = tmp_path / "reviewed.jsonl"
    write_rows(manifest, rows)
    output = tmp_path / "pose"
    assert prepare(manifest, output) == {"train": 1, "val": 1, "test": 0}
    assert [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()] == rows
    result = public_validate(output)
    assert result["valid"], result
    assert result["source_kinds"] == ["real_photo"]
    dataset = json.loads((output / "dataset.json").read_text())
    assert dataset["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len(dataset["files"]) == 2
    rows[0]["consent"] = "catalog_research_only"
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="consent"): prepare(manifest, tmp_path / "no-consent")


def test_content_fingerprint_changes_when_image_or_label_bytes_change(tmp_path):
    _, _, output, _ = exported(tmp_path)
    before = public_validate(output)
    label = next((output / "labels/train").glob("*.txt"))
    label.write_bytes(label.read_bytes() + b"\n")
    after = public_validate(output)
    assert before["manifest_sha256"] == after["manifest_sha256"]
    assert before["content_sha256"] != after["content_sha256"]
    assert not after["valid"]


def test_validation_fails_closed_for_bad_manifest_and_noncanonical_export_id(tmp_path):
    _, _, output, _ = exported(tmp_path)
    manifest = output / "manifest.jsonl"
    original = manifest.read_text()
    manifest.write_text("{broken json")
    assert not public_validate(output)["valid"]
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["export_id"] = "../../escape"
    write_rows(manifest, rows)
    assert not public_validate(output)["valid"]


def test_ultralytics_resolves_exactly_verified_image_and_label_roots(tmp_path, monkeypatch):
    from ultralytics.data import utils
    _, _, output, _ = exported(tmp_path)
    monkeypatch.setattr(utils, "check_font", lambda *args, **kwargs: None)
    data = utils.check_det_dataset(str(output / "pose.yaml"), autodownload=False)
    assert Path(data["path"]) == output
    for split in SPLITS:
        assert Path(data[split]) == output / "images" / split
        images = list((output / "images" / split).glob("*.jpg"))
        labels = utils.img2label_paths(images)
        assert set(map(Path, labels)) == set((output / "labels" / split).glob("*.txt"))


@pytest.mark.parametrize("link_type", ["image_hardlink", "label_hardlink", "yaml_hardlink", "image_symlink", "directory_symlink"])
def test_validator_rejects_linked_files_and_directories(tmp_path, link_type):
    _, _, output, _ = exported(tmp_path)
    if link_type.startswith("image"):
        path = next((output / "images/train").glob("*.jpg"))
    elif link_type.startswith("label"):
        path = next((output / "labels/train").glob("*.txt"))
    elif link_type.startswith("yaml"):
        path = output / "pose.yaml"
    else:
        path = output / "images/train"
    outside = tmp_path / "linked-source"
    if path.is_dir():
        path.rename(outside)
    else:
        outside.write_bytes(path.read_bytes())
        path.unlink()
    try:
        if link_type.endswith("hardlink"):
            path.hardlink_to(outside)
        else:
            path.symlink_to(outside, target_is_directory=outside.is_dir())
    except OSError as exc:
        pytest.skip("Filesystem does not support creating this link: " + str(exc))
    result = public_validate(output)
    assert not result["valid"]
    assert any("link" in error for error in result["errors"])


def test_validator_checks_provenance_hash_and_consent_on_existing_exports(tmp_path):
    _, _, output, _ = exported(tmp_path)
    path = output / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["source_kind"] = "real_photo"
    rows[0]["source"] = "owner-photo"
    rows[0]["session_id"] = "photo-session"
    write_rows(path, rows)
    result = public_validate(output)
    assert not result["valid"]
    assert any("consent" in error for error in result["errors"])


@pytest.mark.parametrize("corruption", ["unreadable", "label_count", "reverse_winding", "concave", "source_checksum", "duplicate_jpeg", "missing_split"])
def test_source_failures_never_create_an_output_directory(tmp_path, corruption):
    source, rows = scenes(tmp_path)
    image = source / "images/train/train-0.jpg"
    if corruption == "unreadable": image.write_bytes(b"not jpeg")
    elif corruption == "label_count": rows[0]["objects"].pop()
    elif corruption == "reverse_winding": rows[0]["objects"][0]["corners"] = QUAD[::-1]
    elif corruption == "concave": rows[0]["objects"][0]["corners"] = [[20, 10], [180, 10], [60, 30], [20, 90]]
    elif corruption == "source_checksum": rows[0]["sha256"] = "wrong"
    elif corruption == "duplicate_jpeg":
        other = source / "images/val/val-0.jpg"
        other.write_bytes(image.read_bytes())
        rows[2]["sha256"] = rows[0]["sha256"]
    else: rows = [row for row in rows if row["split"] != "val"]
    write_rows(source / "manifest.jsonl", rows)
    with pytest.raises(ValueError): prepare_scenes(source, tmp_path / "no-output")
    assert not (tmp_path / "no-output").exists()


def test_export_never_overwrites_existing_dataset(tmp_path):
    source, rows = scenes(tmp_path)
    output = tmp_path / "pose"
    output.mkdir()
    sentinel = output / "owner-file"
    sentinel.write_bytes(b"keep me")
    with pytest.raises(ValueError, match="exists"): prepare_scenes(source, output)
    assert sentinel.read_bytes() == b"keep me"
    assert list(output.iterdir()) == [sentinel]


def test_photo_relative_paths_and_rotated_semantic_corners_are_supported(tmp_path):
    source = tmp_path / "photos"
    source.mkdir()
    image = source / "card.png"
    Image.new("RGB", (200, 100), "orange").save(image)
    row = {"sample_id": "one", "session_id": "one", "card_id": "set:1", "language": "en",
           "side": "front", "variant": "unknown", "source": "owned-photo", "source_kind": "real_photo",
           "consent": "opt_in_training", "path": "card.png", "split": "train", "corners": QUAD[2:] + QUAD[:2]}
    manifest = source / "manifest.jsonl"
    write_rows(manifest, [row])
    assert prepare(manifest, tmp_path / "pose")["train"] == 1
    label = next((tmp_path / "pose/labels/train").glob("*.txt"))
    values = list(map(float, label.read_text().split()))
    assert values[5:7] == pytest.approx([.9, .9])


def test_unassigned_photo_splits_keep_art_sessions_cards_and_duplicates_together():
    from ml.prepare_pose import _split_rows
    rows = [
        {"sample_id": "a", "session_id": "one", "physical_card_id": "card-one", "corners": QUAD},
        {"sample_id": "b", "session_id": "two", "physical_card_id": "card-one", "corners": QUAD},
        {"sample_id": "c", "session_id": "two", "physical_card_id": "card-two", "corners": QUAD},
        {"sample_id": "d", "session_id": "three", "artwork_id": "shared", "physical_card_id": "card-two", "corners": QUAD},
        {"sample_id": "e", "session_id": "four", "group": "shared", "corners": QUAD},
        {"sample_id": "f", "session_id": "five", "corners": QUAD},
    ]
    result = _split_rows(rows, ["a", "b", "c", "d", "same-image", "same-image"])
    assert len({row["split"] for row in result}) == 1


def test_validator_rejects_duplicate_objects_that_ultralytics_deduplicates(tmp_path):
    _, _, output, _ = exported(tmp_path, negative=False)
    manifest = output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    row = rows[0]
    row["objects"][1] = dict(row["objects"][0])
    write_rows(manifest, rows)
    name = hashlib.sha256(row["sample_id"].encode()).hexdigest()
    label = output / "labels/train" / (name + ".txt")
    first = label.read_text().splitlines()[0]
    label.write_text(first + "\n" + first)
    # Remove optional export checksums to isolate actual loader/label semantics.
    (output / "dataset.json").unlink()
    row.pop("pose_label_sha256", None)
    write_rows(manifest, rows)
    result = public_validate(output)
    assert not result["valid"]
    assert any("duplicate_pose_label" in error for error in result["errors"])


def test_source_rejects_jpeg_bytes_ultralytics_would_rewrite(tmp_path):
    source, rows = scenes(tmp_path)
    path = source / "images/train/train-0.jpg"
    path.write_bytes(path.read_bytes() + b"trailing data")
    rows[0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_rows(source / "manifest.jsonl", rows)
    with pytest.raises(ValueError, match="jpeg"):
        prepare_scenes(source, tmp_path / "pose")
    assert not (tmp_path / "pose").exists()


def test_source_rejects_duplicate_pose_geometry_before_writing(tmp_path):
    source, rows = scenes(tmp_path)
    rows[0]["objects"][1] = dict(rows[0]["objects"][0])
    label = source / "labels/train/train-0.txt"
    label.write_text("0 .5 .5 .8 .8\n0 .5 .5 .8 .8")
    rows[0]["label_sha256"] = hashlib.sha256(label.read_bytes()).hexdigest()
    write_rows(source / "manifest.jsonl", rows)
    with pytest.raises(ValueError, match="duplicate_pose_label"):
        prepare_scenes(source, tmp_path / "pose")
    assert not (tmp_path / "pose").exists()


@pytest.mark.parametrize("kind", [[], {}, 7])
def test_malformed_object_source_kinds_return_errors(tmp_path, kind):
    _, _, output, _ = exported(tmp_path)
    path = output / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["objects"][0]["source_kind"] = kind
    write_rows(path, rows)
    assert not public_validate(output)["valid"]


@pytest.mark.parametrize("field,value", [("kpt_shape", [4.0, 3.0]), ("flip_idx", [1.0, 0.0, 3.0, 2.0]), ("names", {False: "pokemon_card"}), ("nc", True), ("channels", 3.0)])
def test_yaml_numeric_schema_types_must_be_usable_by_ultralytics(tmp_path, field, value):
    _, _, output, _ = exported(tmp_path)
    # YAML retains boolean keys while JSON stringifies them.
    import yaml
    path = output / "pose.yaml"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(yaml.safe_dump(config))
    assert not public_validate(output)["valid"]


def test_label_errors_keep_public_diagnostic_prefix(tmp_path):
    _, _, output, _ = exported(tmp_path)
    path = next((output / "labels/train").glob("*.txt"))
    path.write_text("0 .5 .5 .8 .8 .1 .1 2 .9 .1 2 .9 .9 2")
    assert any(error.startswith("invalid_pose_label:") for error in public_validate(output)["errors"])


def test_reviewed_manifest_synthetic_jpegs_keep_exact_source_bytes(tmp_path):
    image = tmp_path / "synthetic.jpg"
    Image.new("RGB", (200, 100), "purple").save(image, quality=80)
    row = {"sample_id": "one", "session_id": "one", "card_id": "set:1", "language": "en", "side": "front",
           "variant": "unknown", "source": "card.webp", "source_kind": "synthetic_catalog_transform",
           "consent": "catalog_research_only", "path": str(image), "split": "train", "corners": QUAD}
    manifest = tmp_path / "manifest.jsonl"
    write_rows(manifest, [row])
    output = tmp_path / "pose"
    prepare(manifest, output)
    assert next((output / "images/train").glob("*.jpg")).read_bytes() == image.read_bytes()


@pytest.mark.parametrize("format", ["PNG", "WEBP"])
def test_validator_supports_older_lossless_image_files(tmp_path, format):
    _, _, output, _ = exported(tmp_path, negative=False)
    (output / "dataset.json").unlink()
    path = output / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        name = hashlib.sha256(row["sample_id"].encode()).hexdigest()
        image = output / "images" / row["split"] / (name + ".jpg")
        replacement = image.with_suffix("." + format.lower())
        with Image.open(image) as pixels: pixels.save(replacement, format=format)
        image.unlink()
        row["sha256"] = hashlib.sha256(replacement.read_bytes()).hexdigest()
    write_rows(path, rows)
    result = public_validate(output)
    assert result["valid"], result


def test_validator_checks_actual_label_quad_even_inside_manifest_tolerance(tmp_path):
    _, _, output, _ = exported(tmp_path, negative=False)
    (output / "dataset.json").unlink()
    manifest = output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    row = rows[0]
    row["objects"] = [{"corners": [[100, 50], [100.001, 50], [100.001, 50.001], [100, 50.001]]}]
    row.pop("pose_label_sha256", None)
    write_rows(manifest, rows)
    name = hashlib.sha256(row["sample_id"].encode()).hexdigest()
    label = output / "labels/train" / (name + ".txt")
    label.write_text("0 .5000025 .500005 .000005 .00001 .5 .5 2 .5 .5 2 .5 .5 2 .5 .5 2")
    result = public_validate(output)
    assert not result["valid"]
    assert any("corners" in error for error in result["errors"])
