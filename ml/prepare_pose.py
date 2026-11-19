"""Export reviewed photo corners; scene exports and validation share one contract."""
import argparse
import io
from pathlib import Path

from ml.pose_dataset import (
    _consent, _groups, _image_size, _leakage, _linked, _name, _objects, _rows, _sha,
    _unique_labels, _values, _write_dataset, validate_pose_dataset,
)


def _split_rows(rows, digests):
    """Connect all declared art/session/physical/duplicate-image groups first."""
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, (row, digest) in enumerate(zip(rows, digests)):
        for key in (*_groups(row), "image:" + digest):
            owner = owners.setdefault(key, index)
            parents[root(index)] = root(owner)
    components = {}
    for index, row in enumerate(rows):
        components.setdefault(root(index), []).append(row["sample_id"])
    splits = {}
    for group, samples in components.items():
        bucket = int(_sha(min(samples).encode())[:8], 16) % 10
        splits[group] = "test" if bucket == 0 else "val" if bucket == 1 else "train"
    return [{**row, "split": splits[root(index)]} for index, row in enumerate(rows)]


def prepare(manifest, output):
    """Keep source metadata verbatim and put exported-file checksums in dataset.json."""
    from cardlens_ai.imaging import decode

    manifest, output = Path(manifest), Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_junction():
        raise ValueError("Output exists; do not overwrite a prepared dataset")
    rows = _rows(manifest)
    assigned = ["split" in row for row in rows]
    if any(assigned) and not all(assigned):
        raise ValueError("Do not mix frozen and unassigned splits")
    prepared, seen = [], set()
    for row in rows:
        for field in ("sample_id", "session_id", "card_id", "language", "side", "variant", "source", "consent", "path"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError("Missing field: " + field)
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample_id")
        seen.add(row["sample_id"])
        if row["side"] not in ("front", "back"):
            raise ValueError("Unknown card side")
        if not _objects(row):
            raise ValueError("All pose samples need reviewed TL/TR/BR/BL corners")
        _consent(row)
        image_path = Path(row["path"])
        if not image_path.is_absolute() and not image_path.is_file():
            image_path = manifest.resolve().parent / image_path
        if _linked(image_path) or not image_path.is_file():
            raise ValueError("Missing image or linked source")
        raw = image_path.read_bytes()
        digest = _sha(raw)
        if row.get("sha256", digest) != digest:
            raise ValueError("Image checksum mismatch")
        image = decode(raw)
        labels = [" ".join(map(str, _values(obj, *image.size))) for obj in _objects(row)]
        _unique_labels(labels)
        copy_bytes = row.get("source_kind", "").startswith("synthetic_") and raw.startswith(b"\xff\xd8")
        if copy_bytes:
            _image_size(raw, ".jpg")
            image_digest = digest
        else:
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=95)
            image_digest = _sha(encoded.getvalue())
        prepared.append({"path": image_path, "source_sha256": digest, "copy_bytes": copy_bytes,
                         "image_sha256": image_digest, "labels": labels})
    digests = [item["source_sha256"] for item in prepared]
    if not all(assigned):
        rows = _split_rows(rows, digests)
    for row in rows:
        _name(row)
    _leakage(rows, digests)
    _leakage(rows, [item["image_sha256"] for item in prepared])
    summary = _write_dataset(rows, prepared, output.resolve(), _sha(manifest.read_bytes()), scenes=False)
    return summary["counts"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(prepare(args.manifest, args.output))
