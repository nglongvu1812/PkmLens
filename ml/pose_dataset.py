"""One pose export/validation contract for reviewed photos and synthetic scenes.

Corners are semantic TL/TR/BR/BL on the card, not sorted by image position.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


SPLITS = ("train", "val", "test")
KEYPOINTS = ["top_left", "top_right", "bottom_right", "bottom_left"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
GROUP_FIELDS = ("group", "art_group", "artwork_group", "artwork_id", "source_group",
                "card_id", "session_id", "physical_card_id")
CONSENTS = ("opt_in_training", "research_and_evaluation")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _rows(path):
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("pose_manifest_invalid") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("pose_manifest_empty_or_invalid")
    return rows


def _linked(path):
    return (path.is_symlink() or path.is_junction() or
            (path.is_file() and path.stat().st_nlink > 1))


def _local(root, relative):
    path = root / relative
    if not path.is_relative_to(root) or ".." in Path(relative).parts:
        raise ValueError("pose_path_outside_dataset")
    for part in (path, *path.parents):
        if _linked(part):
            raise ValueError("pose_links_not_allowed:" + str(path))
        if part == root:
            break
    if not path.resolve().is_relative_to(root):
        raise ValueError("pose_path_outside_dataset")
    return path


def _objects(row):
    objects = row.get("objects", [row] if row.get("corners") is not None else None)
    if not isinstance(objects, list) or any(not isinstance(obj, dict) for obj in objects):
        raise ValueError("pose_manifest_objects_invalid")
    return objects


def _name(row):
    sample = row.get("sample_id")
    if not isinstance(sample, str) or not sample.strip() or row.get("split") not in SPLITS:
        raise ValueError("invalid_pose_manifest_row")
    name = _sha(sample.encode())
    if "export_id" in row and row["export_id"] != name:
        raise ValueError("invalid_pose_export_id")
    return name


def _consent(row):
    kind = row.get("source_kind", "unspecified")
    if not isinstance(kind, str) or not kind:
        raise ValueError("pose_source_kind_invalid")
    for item in (row, *_objects(row)):
        item_kind = item.get("source_kind", kind)
        if not isinstance(item_kind, str) or not item_kind:
            raise ValueError("pose_source_kind_invalid")
        if item_kind == "real_photo":
            if item.get("consent", row.get("consent")) not in CONSENTS:
                raise ValueError("Real-photo usage requires explicit scoped consent")
            if not item.get("session_id", row.get("session_id")) or not item.get("source", row.get("source")):
                raise ValueError("pose_real_photo_source_and_session_required")


def _groups(row):
    for item in (row, *_objects(row)):
        for field in GROUP_FIELDS:
            value = item.get(field)
            if value is not None and value != "":
                if not isinstance(value, str):
                    raise ValueError("pose_source_group_invalid")
                # Different art-field spellings still describe the same grouping namespace.
                namespace = "art" if field in GROUP_FIELDS[:5] else field
                yield namespace + ":" + value
    for obj in _objects(row):
        if obj is not row and obj.get("source"):
            if not isinstance(obj["source"], str):
                raise ValueError("pose_source_group_invalid")
            yield "art_source:" + obj["source"].replace("\\", "/")


def _leakage(rows, digests):
    groups = {}
    for row, digest in zip(rows, digests):
        for key in (*_groups(row), "image:" + digest):
            previous = groups.setdefault(key, row["split"])
            if previous != row["split"]:
                reason = "pose_duplicate_image_across_splits:" if key.startswith("image:") else "pose_source_group_leakage:"
                raise ValueError(reason + key)


def _quad(corners, width, height):
    points = np.asarray(corners, dtype=np.float64)
    if (points.shape != (4, 2) or not np.isfinite(points).all() or np.any(points < 0)
            or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height)):
        raise ValueError("invalid_pose_manifest_corners")
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not np.all(cross > 1e-8):
        raise ValueError("pose_corners_must_be_convex_clockwise_semantic_TL_TR_BR_BL")
    return points / [width, height]


def _values(obj, width, height):
    points = _quad(obj.get("corners"), width, height)
    low, high = points.min(axis=0), points.max(axis=0)
    return np.array([0, *((low + high) / 2), *(high - low),
                     *(value for point in points for value in (*point, 2))])


def _numbers(text, size, reason):
    try:
        values = np.asarray(text.split(), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(reason) from exc
    if values.shape != (size,) or not np.isfinite(values).all() or values[0] != 0:
        raise ValueError(reason)
    if (np.any(values[1:5] < 0) or np.any(values[1:5] > 1) or np.any(values[3:5] <= 0)
            or np.any(values[1:3] - values[3:5] / 2 < -1e-6)
            or np.any(values[1:3] + values[3:5] / 2 > 1 + 1e-6)):
        raise ValueError(reason)
    if size == 17 and (np.any(values[5::3] < 0) or np.any(values[5::3] > 1)
                       or np.any(values[6::3] < 0) or np.any(values[6::3] > 1)
                       or not np.all(values[7::3] == 2)):
        raise ValueError(reason)
    return values


def _unique_labels(lines, reason="invalid_pose_label"):
    # Ultralytics parses float32 and silently drops identical annotation rows.
    values = [_numbers(line, 17, reason) for line in lines]
    if len({tuple(row.astype(np.float32)) for row in values}) != len(values):
        raise ValueError("duplicate_pose_label")


def _image_size(raw, suffix):
    if suffix in (".jpg", ".jpeg") and not raw.endswith(b"\xff\xd9"):
        raise ValueError("pose_jpeg_requires_repair_export_would_change_bytes")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            expected_format = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}[suffix]
            if image.format != expected_format or image.getexif().get(274, 1) != 1:
                raise ValueError("pose_image_format_or_orientation_invalid")
            if min(image.size) < 10 or image.width * image.height > 16_000_000 or getattr(image, "n_frames", 1) != 1:
                raise ValueError("pose_image_dimensions_invalid")
            size = image.size
            image.load()
        return size
    except (OSError, Image.DecompressionBombError, KeyError) as exc:
        raise ValueError("invalid_pose_image") from exc


def _inventory(root):
    files = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            if _linked(path):
                raise ValueError("pose_links_not_allowed:" + str(path.relative_to(root)))
            if path.suffix.lower() in (".cache", ".npy", ".npz"):
                raise ValueError("pose_loader_cache_not_allowed:" + str(path.relative_to(root)))
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                files.append(path)
            else:
                raise ValueError("pose_regular_files_required")
    if (root / "masks").exists():
        raise ValueError("pose_masks_not_allowed")
    return files


def _config(root):
    try:
        config = yaml.safe_load(_local(root, "pose.yaml").read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        raise ValueError("pose_yaml_invalid") from exc
    allowed = {"path", "train", "val", "test", "kpt_shape", "flip_idx", "names", "kpt_names", "nc", "channels"}
    if (not isinstance(config, dict) or set(config) - allowed or config.get("kpt_shape") != [4, 3]
            or config.get("flip_idx") != [1, 0, 3, 2]
            or config.get("names") not in ({0: "pokemon_card"}, {"0": "pokemon_card"})
            or config.get("nc", 1) != 1 or config.get("channels", 3) != 3
            or config.get("kpt_names", {"0": KEYPOINTS}) not in ({0: KEYPOINTS}, {"0": KEYPOINTS})):
        raise ValueError("pose_dataset_schema_invalid")
    if (any(type(value) is not int for value in config["kpt_shape"] + config["flip_idx"])
            or type(config.get("nc", 1)) is not int or type(config.get("channels", 3)) is not int
            or type(next(iter(config["names"]))) not in (str, int)):
        raise ValueError("pose_dataset_schema_invalid")
    path = config.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute() or Path(path).resolve() != root or _linked(Path(path)):
        raise ValueError("pose_yaml_root_must_match_dataset")
    for split in SPLITS:
        if config.get(split) is None and split == "test":
            continue
        if config.get(split) != "images/" + split:
            raise ValueError("pose_split_path_invalid:" + split)
        if not _local(root, config[split]).is_dir() or not _local(root, "labels/" + split).is_dir():
            raise ValueError("pose_split_path_invalid:" + split)
    return config


def validate_pose_dataset(root):
    """Return a fail-closed report for exactly the files consumed by pose.yaml.

    Persistent Ultralytics label/NPY caches are rejected because the loader can
    use them instead of the image/label bytes inspected here. Validation is
    tuning evidence; only a separately declared test split is a held-out test.
    """
    root = Path(root).absolute()
    errors, counts, warnings, fingerprints = [], {}, [], {}
    kinds = set()
    result = {"valid": False, "errors": errors, "counts": counts, "manifest_sha256": None,
              "source_kinds": [], "content_sha256": None, "warnings": warnings,
              "split_roles": {"train": "training", "val": "validation (model selection; not a test)"}}

    def fingerprint(path):
        digest = _sha(path.read_bytes())
        fingerprints[path.relative_to(root).as_posix()] = digest
        return digest

    try:
        if _linked(root) or not root.is_dir():
            raise ValueError("pose_dataset_root_invalid")
        root = root.resolve()
        inventory = _inventory(root)
        for name in ("pose.yaml", "manifest.jsonl", "dataset.json"):
            path = root / name
            if path.is_file():
                fingerprint(path)
        result["manifest_sha256"] = fingerprints.get("manifest.jsonl")
        config = _config(root)
        splits = [split for split in SPLITS if config.get(split) is not None]
        if "test" in splits:
            result["split_roles"]["test"] = "held-out test (separate from validation)"
        else:
            warnings.append("No held-out test split; validation results are not test results.")
        manifest_rows = _rows(_local(root, "manifest.jsonl"))
        by_name = {}
        for row in manifest_rows:
            name = _name(row)
            if name in by_name:
                raise ValueError("pose_duplicate_sample_id:" + row["sample_id"])
            if row["split"] not in splits:
                raise ValueError("pose_manifest_split_not_declared:" + row["split"])
            _consent(row)
            kinds.add(row.get("source_kind", "unspecified"))
            kinds.update(obj["source_kind"] for obj in _objects(row) if obj.get("source_kind"))
            by_name[name] = row
        metadata = {}
        if (root / "dataset.json").is_file():
            metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8-sig"))
            if not isinstance(metadata, dict):
                raise ValueError("pose_dataset_metadata_invalid")
            if metadata.get("prepared_manifest_sha256", result["manifest_sha256"]) != result["manifest_sha256"]:
                errors.append("pose_prepared_manifest_checksum_mismatch")
        file_records = metadata.get("files")
        if file_records is not None and (not isinstance(file_records, dict) or set(file_records) != set(by_name)):
            raise ValueError("pose_file_provenance_manifest_mismatch")
        seen, image_rows, image_digests = set(), [], []
        expected_files = set()
        for split in splits:
            image_dir, label_dir = root / "images" / split, root / "labels" / split
            images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            objects_count = 0
            counts[split] = {"images": len(images), "objects": 0}
            for image in images:
                label = label_dir / (image.stem + ".txt")
                expected_files.update((image, label))
                digest = fingerprint(image)
                row = by_name.get(image.stem)
                if row is None or row["split"] != split or image.stem in seen:
                    errors.append("pose_image_label_manifest_mismatch:" + image.name)
                    continue
                seen.add(image.stem)
                image_rows.append(row)
                image_digests.append(digest)
                record = file_records[image.stem] if file_records is not None else {}
                if not isinstance(record, dict):
                    raise ValueError("pose_file_provenance_invalid")
                expected_hash = record.get("image_sha256", row.get("sha256"))
                if expected_hash and expected_hash != digest:
                    errors.append("pose_image_hash_mismatch:" + image.name)
                if (record.get("source_sha256") and row.get("sha256")
                        and record["source_sha256"] != row["sha256"]):
                    errors.append("pose_source_image_hash_mismatch:" + image.name)
                try:
                    label_digest = fingerprint(label)
                    if record.get("label_sha256", row.get("pose_label_sha256", label_digest)) != label_digest:
                        errors.append("pose_label_hash_mismatch:" + image.name)
                    width, height = _image_size(image.read_bytes(), image.suffix.lower())
                    lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
                    _unique_labels(lines, "invalid_pose_label:" + image.name)
                    objects = _objects(row)
                    if len(lines) != len(objects):
                        errors.append("pose_manifest_label_count_mismatch:" + image.name)
                    for index, line in enumerate(lines):
                        values = _numbers(line, 17, "invalid_pose_label:" + image.name)
                        label_corners = np.column_stack((values[5::3], values[6::3])) * [width, height]
                        _quad(label_corners, width, height)
                        if index >= len(objects):
                            continue
                        expected = _values(objects[index], width, height)
                        if not np.allclose(values, expected, atol=1e-5, rtol=0):
                            errors.append("pose_label_manifest_corner_or_box_mismatch:" + image.name)
                            continue
                        objects_count += 1
                except (OSError, ValueError, TypeError, UnicodeError) as exc:
                    errors.append(str(exc) if isinstance(exc, ValueError) else "invalid_pose_image_or_label:" + image.name)
            counts[split]["objects"] = objects_count
            if not images or (split != "test" and not objects_count):
                errors.append("empty_pose_split:" + split)
        for path in inventory:
            relative = path.relative_to(root)
            if relative.parts[0] in ("images", "labels"):
                fingerprint(path)
                if path not in expected_files or len(relative.parts) != 3:
                    errors.append("pose_unverified_image_or_label:" + relative.as_posix())
        if seen != set(by_name):
            errors.append("pose_manifest_image_count_mismatch")
        try:
            _leakage(image_rows, image_digests)
        except ValueError as exc:
            errors.append(str(exc))
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        errors.append(str(exc))
    result["source_kinds"] = sorted(kinds)
    result["content_sha256"] = _sha(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode())
    result["valid"] = not errors
    result["errors"] = errors[:100]
    return result


def _write_dataset(rows, prepared, output, source_manifest_sha256, *, scenes):
    """Write only after all sources, geometry, checksums and splits pass preflight."""
    output.mkdir(parents=True, exist_ok=False)
    present_splits = {row["split"] for row in rows} | {"train", "val"}
    for split in SPLITS:
        if split in present_splits:
            (output / "images" / split).mkdir(parents=True)
            (output / "labels" / split).mkdir(parents=True)
    exported, file_records = [], {}
    for row, item in zip(rows, prepared):
        name, split = _name(row), row["split"]
        raw = item["path"].read_bytes()
        if _sha(raw) != item["source_sha256"]:
            raise ValueError("pose_source_changed_during_export")
        if not scenes and not item.get("copy_bytes"):
            from cardlens_ai.imaging import decode
            encoded = io.BytesIO()
            decode(raw).save(encoded, format="JPEG", quality=95)
            raw = encoded.getvalue()
        image_path = output / "images" / split / (name + ".jpg")
        label_path = output / "labels" / split / (name + ".txt")
        label_raw = ("\n".join(item["labels"]) + ("\n" if item["labels"] else "")).encode()
        image_path.write_bytes(raw)
        label_path.write_bytes(label_raw)
        file_records[name] = {"source_sha256": item["source_sha256"], "image_sha256": _sha(raw),
                              "label_sha256": _sha(label_raw)}
        if scenes:
            exported.append({**row, "export_id": name, "sha256": _sha(raw),
                             "bbox_label_sha256": item["bbox_label_sha256"], "pose_label_sha256": _sha(label_raw)})
        else:
            exported.append(row)
    config = {"path": str(output), "train": "images/train", "val": "images/val",
              "kpt_shape": [4, 3], "flip_idx": [1, 0, 3, 2], "names": {"0": "pokemon_card"},
              "kpt_names": {"0": KEYPOINTS}}
    if "test" in present_splits:
        config["test"] = "images/test"
    (output / "pose.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest_raw = "\n".join(json.dumps(row) for row in exported).encode()
    (output / "manifest.jsonl").write_bytes(manifest_raw)
    counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    kinds = sorted({item.get("source_kind", row.get("source_kind", "unspecified"))
                    for row in rows for item in (row, *_objects(row))})
    real_count = sum(any(item.get("source_kind", row.get("source_kind")) == "real_photo"
                         for item in (row, *_objects(row))) for row in rows)
    summary = {"kind": "synthetic_only" if all(kind.startswith("synthetic_") for kind in kinds) else "reviewed_photos" if kinds == ["real_photo"] else "mixed_or_unspecified",
               "source_manifest_sha256": source_manifest_sha256, "counts": counts,
               "real_photo_count": real_count, "source_kinds": kinds,
               "prepared_manifest_sha256": _sha(manifest_raw), "files": file_records}
    if not scenes:
        summary["manifest_sha256"] = source_manifest_sha256
    (output / "dataset.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation = validate_pose_dataset(output)
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if scenes and not validation["valid"]:
        raise ValueError(validation)
    return {key: summary[key] for key in ("kind", "source_manifest_sha256", "counts", "real_photo_count")} if scenes else summary


def prepare_scenes(source, output):
    """Export bbox scene geometry without recompressing any synthetic JPEG."""
    source, output = Path(source).absolute(), Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_junction():
        raise ValueError("pose_dataset_exists_choose_new_name")
    if _linked(source):
        raise ValueError("pose_links_not_allowed")
    source, output = source.resolve(), output.resolve()
    manifest = _local(source, "manifest.jsonl")
    if not manifest.is_file():
        raise ValueError("scene_manifest_missing")
    rows = _rows(manifest)
    prepared, seen = [], set()
    for row in rows:
        name = _name(row)
        sample, split = row["sample_id"], row["split"]
        if any(character in sample for character in ("/", "\\", ":")) or sample in (".", ".."):
            raise ValueError("invalid_scene_sample_id")
        if name in seen:
            raise ValueError("pose_duplicate_sample_id:" + sample)
        seen.add(name)
        image = _local(source, f"images/{split}/{sample}.jpg")
        label = _local(source, f"labels/{split}/{sample}.txt")
        if not image.is_file():
            raise ValueError("scene_image_missing_or_outside_dataset")
        if not label.is_file():
            raise ValueError("scene_bbox_label_missing")
        raw, label_raw = image.read_bytes(), label.read_bytes()
        digest, label_digest = _sha(raw), _sha(label_raw)
        if row.get("sha256", digest) != digest:
            raise ValueError("scene_image_checksum_mismatch")
        if row.get("label_sha256", label_digest) != label_digest:
            raise ValueError("scene_label_checksum_mismatch")
        width, height = _image_size(raw, ".jpg")
        objects = _objects(row)
        lines = [line for line in label_raw.decode("utf-8-sig").splitlines() if line.strip()]
        if len(lines) != len(objects):
            raise ValueError("scene_manifest_bbox_count_mismatch")
        labels = []
        for line, obj in zip(lines, objects):
            bbox = _numbers(line, 5, "invalid_scene_bbox_label")
            expected = _values(obj, width, height)
            if not np.allclose(expected[:5], bbox, atol=1e-5, rtol=0):
                raise ValueError("scene_box_does_not_match_pose_corners")
            labels.append(" ".join(map(str, expected)))
        _unique_labels(labels)
        _consent(row)
        prepared.append({"path": image, "source_sha256": digest, "bbox_label_sha256": label_digest, "labels": labels})
    _leakage(rows, [item["source_sha256"] for item in prepared])
    for split in ("train", "val"):
        if not any(row["split"] == split and item["labels"] for row, item in zip(rows, prepared)):
            raise ValueError("empty_pose_split:" + split)
    return _write_dataset(rows, prepared, output, _sha(manifest.read_bytes()), scenes=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.action == "prepare" and args.output is None:
        parser.error("prepare requires <source-dir> <output-dir>")
    result = validate_pose_dataset(args.source) if args.action == "validate" else prepare_scenes(args.source, args.output)
    print(json.dumps(result, indent=2))
    if args.action == "validate" and not result["valid"]:
        raise SystemExit(1)
