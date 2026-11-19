"""Local four-corner YOLO26 pose training, resume, evaluation and ONNX parity."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

from ml.prepare_pose import validate_pose_dataset
from ml.train_detection import preserve_resume_checkpoint


CLASSES = ["pokemon_card"]
KEYPOINTS = ["TL", "TR", "BR", "BL"]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _local_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.suffix.lower() != ".pt" or not checkpoint.is_file():
        raise ValueError("trusted_local_pt_checkpoint_required")
    return checkpoint


def _yolo():
    # Disable optional network/update and dependency installation before Ultralytics imports.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / "data/yolo-config"))
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics_not_installed") from exc
    return YOLO


@contextmanager
def _uncached_labels():
    # cache=False disables image caching, not label .cache reads/writes in 8.4.214.
    with patch("ultralytics.data.dataset.load_dataset_cache_file", side_effect=FileNotFoundError), \
            patch("ultralytics.data.dataset.save_dataset_cache_file",
                  side_effect=lambda prefix, path, cache, version: cache.update(version=version)):
        yield


def _device(device):
    if device not in ("auto", "cpu", "0"):
        raise ValueError("device_must_be_auto_cpu_or_0")
    if device == "cpu":
        return device
    import torch
    if device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("cuda_unavailable_choose_cpu")
    return device


def _configuration(epochs, batch, imgsz, scale):
    if (type(epochs) is not int or not 1 <= epochs <= 300 or
            type(batch) is not int or not 1 <= batch <= 16 or
            type(imgsz) is not int or imgsz not in (320, 480, 640) or scale not in ("n", "s")):
        raise ValueError("invalid_training_configuration")


def create_pose_model_yaml(output, scale="n"):
    """Write an architecture only; its filename preserves Ultralytics scale inference."""
    import yaml
    _yolo()
    import ultralytics

    if scale not in ("n", "s"):
        raise ValueError("invalid_pose_model_architecture")
    output = Path(output).resolve()
    if output.name != f"yolo26{scale}-card-pose.yaml":
        raise ValueError("pose_architecture_filename_must_preserve_scale")
    if output.exists():
        raise ValueError("pose_architecture_exists_choose_new_run")
    source = Path(ultralytics.__file__).resolve().parent / "cfg/models/26/yolo26-pose.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config.update(scale=scale, nc=1, kpt_shape=[4, 3], names={0: CLASSES[0]},
                  kpt_names={0: ["top_left", "top_right", "bottom_right", "bottom_left"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return output


def _schema(model, fresh=False, scale=None):
    architecture = model.model
    config = getattr(architecture, "yaml", {})
    names = model.names
    valid_names = names in ({0: CLASSES[0]}, {"0": CLASSES[0]}, CLASSES)
    if (model.task != "pose" or list(getattr(architecture, "kpt_shape", [])) != [4, 3] or
            config.get("nc") != 1 or (not fresh and not valid_names) or
            (scale is not None and config.get("scale") != scale)):
        raise ValueError("checkpoint_must_be_four_corner_pose")


def _source_kinds(rows, split=None):
    kinds = set()
    for row in rows:
        if split is None or row.get("split") == split:
            kinds.add(str(row.get("source_kind", "unspecified")))
            kinds.update(str(obj["source_kind"]) for obj in row.get("objects", []) if obj.get("source_kind"))
    return sorted(kinds)


def _dataset(data):
    data = Path(data).resolve()
    if data.name != "pose.yaml" or not data.is_file():
        raise ValueError("reviewed_pose_yaml_required")
    validation = validate_pose_dataset(data.parent)
    if not validation.get("valid"):
        raise ValueError("pose_dataset_invalid:" + json.dumps(validation.get("errors", [])))
    # The shared validator supplies the content hash when available; older versions need a fallback.
    digest = validation.get("dataset_sha256") or validation.get("content_sha256")
    if not digest:
        content = hashlib.sha256()
        paths = [data, data.parent / "manifest.jsonl"]
        for directory, suffixes in (("images", {".jpg", ".jpeg", ".png", ".webp"}), ("labels", {".txt"})):
            paths.extend(path for path in (data.parent / directory).rglob("*")
                         if path.is_file() and path.suffix.lower() in suffixes)
        for path in sorted(paths):
            content.update(path.relative_to(data.parent).as_posix().encode("utf-8") + b"\0")
            content.update(_sha256(path).encode("ascii"))
        digest = content.hexdigest()
    rows = [json.loads(line) for line in (data.parent / "manifest.jsonl").read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    kinds = validation.get("source_kinds") or _source_kinds(rows)
    validation_kinds = _source_kinds(rows, "val")
    return data, {"dataset": str(data.parent), "dataset_sha256": digest,
                  "manifest_sha256": validation["manifest_sha256"], "counts": validation["counts"],
                  "source_kinds": kinds, "validation_source_kinds": validation_kinds,
                  "dataset_kind": "synthetic_only" if kinds and all(k.startswith("synthetic") for k in kinds)
                  else "real_photo" if kinds == ["real_photo"] else "mixed_or_unspecified"}, rows


def _prepare_run(output, expected, resume, checkpoint):
    """Accept an empty/API-bootstrap folder, never previous training artifacts."""
    config_path = output / "pose-config.json"
    previous = _read_json(config_path) if config_path.is_file() else {}
    if resume:
        if not previous or previous.get("task") != "pose":
            raise ValueError("resume_pose_config_required")
        if checkpoint.parent != output / "weights" or not re.fullmatch(r"(?:resume|last|epoch\d+)\.pt", checkpoint.name):
            raise ValueError("resume_checkpoint_must_belong_to_same_run")
        return previous
    if output.exists():
        bootstrap = {"pose-config.json", "pose-status.json", "pose-result.json", "pose-train.log", "STOP"}
        if any(path.name not in bootstrap or path.is_dir() for path in output.iterdir()):
            raise ValueError("pose_run_exists_choose_new_output")
        for filename in ("pose-status.json", "pose-result.json"):
            if (output / filename).is_file():
                state = _read_json(output / filename)
                pending_stop = filename == "pose-status.json" and state.get("status") == "stopping" and (output / "STOP").exists()
                if (state.get("status") != "starting" and not pending_stop) or state.get("validation_passed") or state.get("checkpoint"):
                    raise ValueError("pose_run_exists_choose_new_output")
        if previous:
            for key in ("dataset", "checkpoint", "task", "epochs", "batch", "imgsz", "device", "scale"):
                if key in previous and previous[key] != expected[key]:
                    raise ValueError("pose_run_exists_configuration_mismatch:" + key)
    output.mkdir(parents=True, exist_ok=True)
    return None


def _resume_state(model, output, data, config):
    _schema(model, scale=config["scale"])
    checkpoint = getattr(model, "ckpt", None)
    optimizer = checkpoint.get("optimizer") if isinstance(checkpoint, dict) else None
    epoch = checkpoint.get("epoch", -1) if isinstance(checkpoint, dict) else -1
    if (type(epoch) is not int or not 0 <= epoch < config["epochs"] - 1 or
            not isinstance(optimizer, dict) or not optimizer.get("state") or not optimizer.get("param_groups")):
        raise ValueError("resume_requires_resumable_epoch_and_optimizer_state")
    args = checkpoint.get("train_args", {})
    if (args.get("task") != "pose" or not args.get("save_dir") or
            Path(args["save_dir"]).resolve() != output or
            Path(args.get("project", "")).resolve() != output.parent or args.get("name") != output.name):
        raise ValueError("resume_checkpoint_must_belong_to_same_run")
    if not args.get("data") or Path(args["data"]).resolve() != data:
        raise ValueError("resume_checkpoint_dataset_must_match_original")
    if any(args.get(key) != config[key] for key in ("epochs", "batch", "imgsz")):
        raise ValueError("resume_checkpoint_configuration_must_match_original")
    return epoch + 1


def _gate_thresholds():
    thresholds = {}
    for key, default in (("map50", .75), ("precision", .70), ("recall", .70)):
        try:
            value = float(os.environ.get("POSE_MIN_" + key.upper(), default))
        except ValueError as exc:
            raise ValueError("pose_gate_threshold_invalid:" + key) from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("pose_gate_threshold_must_be_finite_in_0_1:" + key)
        thresholds[key] = value
    return thresholds


def _metrics(measured):
    metrics = {str(key): float(value) for key, value in measured.results_dict.items()}
    if not metrics or not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("pose_validation_metrics_not_finite")
    return metrics


def _measurement(model, data, device, batch, imgsz, rows, split="val", output=None, thresholds=None):
    kwargs = dict(data=str(data), split=split, device=device, batch=batch, imgsz=imgsz, workers=0, plots=False)
    if output is not None:
        kwargs.update(project=str(output), name=split, exist_ok=True)
    with _uncached_labels():
        metrics = _metrics(model.val(**kwargs))
    kinds = _source_kinds(rows, split)
    thresholds = thresholds if thresholds is not None else _gate_thresholds()
    passed = all(0 <= metrics.get(metric, -1) <= 1 and metrics[metric] >= thresholds[key]
                 for key, metric in (("map50", "metrics/mAP50(P)"), ("precision", "metrics/precision(P)"),
                                     ("recall", "metrics/recall(P)")))
    return {"metrics": metrics, "split": split, "validation_source_kinds": kinds,
            "split_role": "held_out_test" if split == "test" else "validation_model_selection",
            "gate_thresholds": thresholds, "validation_passed": passed,
            "real_photo_validated": False,
            "validation_scope": ("synthetic_test_only" if split == "test" else "synthetic_validation_only")
            if kinds and all(k.startswith("synthetic") for k in kinds)
            else "dataset_split_only_not_real_camera_validation"}


def train(data, checkpoint, output, epochs=10, batch=2, imgsz=480, device="cpu", resume=False,
          model_yaml=None, callback=None, scale="n"):
    checkpoint, output, data = Path(checkpoint).resolve(), Path(output).resolve(), Path(data).resolve()
    expected = {"dataset": str(data.parent), "checkpoint": str(checkpoint), "task": "pose", "scale": scale,
                "epochs": epochs, "batch": batch, "imgsz": imgsz, "device": device}
    previous = _prepare_run(output, expected, resume, checkpoint)
    # Discard only the prior stop request, before validation; a new startup STOP must survive.
    if resume:
        state_path = output / "pose-status.json"
        pending_stop = state_path.is_file() and _read_json(state_path).get("status") == "stopping"
        if not pending_stop:
            (output / "STOP").unlink(missing_ok=True)
    config = previous or expected
    base = {"status": "starting", "task": "pose", "classes": CLASSES, "keypoints": KEYPOINTS,
            "epoch": 0, "epochs": epochs, "validation_passed": False, "real_photo_validated": False,
            "initial_checkpoint": config["checkpoint"], "dataset": str(data.parent)}
    _write_json(output / "pose-result.json", base)
    _write_json(output / "pose-status.json", base)
    if not resume:
        _write_json(output / "pose-config.json", config)
    epoch = 0
    try:
        _configuration(epochs, batch, imgsz, scale)
        _local_checkpoint(checkpoint)
        thresholds = _gate_thresholds()
        if resume:
            if model_yaml is not None:
                raise ValueError("resume_uses_original_architecture_not_model_yaml")
            for key in ("dataset", "scale", "epochs", "batch", "imgsz"):
                if config.get(key) != expected[key]:
                    raise ValueError("resume_dataset_must_match_original" if key == "dataset" else
                                     "resume_configuration_must_match_original:" + key)
        data, evidence, rows = _dataset(data)
        device = _device(device)
        if resume:
            if any(config.get(key) != evidence[key] for key in ("manifest_sha256", "dataset_sha256")):
                raise ValueError("resume_dataset_content_must_match_original")
        else:
            config.update(evidence, initial_checkpoint_sha256=_sha256(checkpoint),
                          gate_thresholds=thresholds, started_at=time.time())
            _write_json(output / "pose-config.json", config)
        YOLO = _yolo()
        if resume:
            model = YOLO(str(checkpoint), task="pose")
            epoch = _resume_state(model, output, data, config)
        else:
            architecture = output / f"yolo26{scale}-card-pose.yaml"
            if model_yaml is None:
                create_pose_model_yaml(architecture, scale)
            else:
                import yaml
                source = Path(model_yaml).resolve()
                custom = yaml.safe_load(source.read_text(encoding="utf-8"))
                if (not isinstance(custom, dict) or custom.get("nc") != 1 or custom.get("kpt_shape") != [4, 3] or
                        custom.get("names") not in ({0: "pokemon_card"}, {"0": "pokemon_card"}, CLASSES)):
                    raise ValueError("custom_yaml_must_define_four_corner_pokemon_pose")
                custom["scale"] = scale
                with architecture.open("x", encoding="utf-8") as stream:
                    yaml.safe_dump(custom, stream, sort_keys=False)
                config["model_yaml_source"] = str(source)
            config.update(model_yaml=str(architecture), model_yaml_sha256=_sha256(architecture))
            _write_json(output / "pose-config.json", config)
            model = YOLO(str(architecture), task="pose")
            _schema(model, fresh=True, scale=scale)
            model.model.names = {0: CLASSES[0]}
            model.load(str(checkpoint))
        base.update(evidence, initial_checkpoint_sha256=config["initial_checkpoint_sha256"],
                    scale=scale, device=device, batch=batch, imgsz=imgsz)
        saved_epoch = -1
        stop_requested = False

        def on_model_save(trainer):
            nonlocal saved_epoch
            preserve_resume_checkpoint(trainer)
            saved_epoch = trainer.epoch

        def on_epoch_end(trainer):
            nonlocal epoch, stop_requested
            # final_eval emits this event again at epoch+1 without saving an epoch.
            if trainer.epoch != saved_epoch:
                return
            epoch = trainer.epoch + 1
            if (output / "STOP").exists():
                stop_requested = True
                trainer.stop = True
            state = {**base, "status": "stopping" if stop_requested else "training", "epoch": epoch,
                     "metrics": {str(k): float(v) for k, v in trainer.metrics.items()}, "updated_at": time.time()}
            _write_json(output / "pose-status.json", state)
            if callback:
                callback(trainer)

        model.add_callback("on_model_save", on_model_save)
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        with _uncached_labels():
            model.train(data=str(data), epochs=epochs, batch=batch, imgsz=imgsz, device=device,
                        fliplr=0.0, flipud=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                        perspective=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                        resume=resume, project=str(output.parent), name=output.name, save_dir=str(output),
                        exist_ok=True, seed=42, workers=0, cache=False, amp=False, patience=0,
                        save=True, save_period=1, plots=True)
        checkpoint_out = output / "weights/best.pt"
        if not checkpoint_out.is_file():
            checkpoint_out = output / "weights/last.pt"
        _local_checkpoint(checkpoint_out)
        measured = YOLO(str(checkpoint_out), task="pose")
        _schema(measured)
        result = {**base, **_measurement(measured, data, device, batch, imgsz, rows, output=output / "evaluation", thresholds=thresholds),
                  "status": "stopped" if stop_requested or (output / "STOP").exists() else "completed",
                  "epoch": epoch, "checkpoint": str(checkpoint_out), "sha256": _sha256(checkpoint_out),
                  "resume_checkpoint": str(output / "weights/resume.pt"), "finished_at": time.time()}
        _write_json(output / "pose-result.json", result)
        _write_json(output / "pose-status.json", result)
        return result
    except Exception as exc:
        failed = {**base, "status": "failed", "epoch": epoch, "error": str(exc), "validation_passed": False}
        _write_json(output / "pose-result.json", failed)
        _write_json(output / "pose-status.json", failed)
        raise


def evaluate(data, checkpoint, output, split="val", batch=2, imgsz=480, device="cpu"):
    if split not in ("val", "test"):
        raise ValueError("pose_evaluation_split_must_be_val_or_test")
    _configuration(1, batch, imgsz, "n")
    checkpoint = _local_checkpoint(checkpoint)
    thresholds = _gate_thresholds()
    data, evidence, rows = _dataset(data)
    if not evidence["counts"].get(split, {}).get("images"):
        raise ValueError("pose_evaluation_split_missing:" + split)
    device, output = _device(device), Path(output).resolve()
    if output.exists():
        raise ValueError("pose_evaluation_output_exists_choose_new_output")
    model = _yolo()(str(checkpoint), task="pose")
    _schema(model)
    output.mkdir(parents=True)
    result = {"status": "completed", "task": "pose", "classes": CLASSES, "keypoints": KEYPOINTS,
              **evidence, **_measurement(model, data, device, batch, imgsz, rows, split, output, thresholds),
              "checkpoint": str(checkpoint), "sha256": _sha256(checkpoint), "device": device,
              "batch": batch, "imgsz": imgsz, "finished_at": time.time()}
    _write_json(output / "pose-evaluation.json", result)
    return result



def _compare_pose_results(before, after):
    """Match card boxes one-to-one before comparing their ordered corner predictions."""
    import numpy as np

    values = []
    for result in (before, after):
        if getattr(result, "boxes", None) is None or getattr(result, "keypoints", None) is None:
            raise ValueError("boxes_and_four_corner_confidences_required")
        boxes = result.boxes.data.cpu().numpy()
        points = result.keypoints.data.cpu().numpy()
        if (boxes.shape != (len(boxes), 6) or points.shape != (len(boxes), 4, 3) or
                not np.isfinite(boxes).all() or not np.isfinite(points).all()):
            raise ValueError("invalid_or_nonfinite_pose_output")
        if (np.any(boxes[:, 5] != 0) or np.any(boxes[:, 2:4] <= boxes[:, :2]) or
                np.any((boxes[:, 4] < 0) | (boxes[:, 4] > 1)) or
                np.any((points[:, :, 2] < 0) | (points[:, :, 2] > 1))):
            raise ValueError("invalid_card_boxes_or_confidences")
        values.append((boxes, points))
    (x, u), (y, v) = values
    if len(x) != len(y):
        raise ValueError("detection_count_differs")
    matches = np.empty(len(x), dtype=int)
    matched_ious = []
    if len(x):
        intersection = np.maximum(0, np.minimum(x[:, None, 2:4], y[None, :, 2:4]) -
                                  np.maximum(x[:, None, :2], y[None, :, :2])).prod(axis=2)
        area_x = (x[:, 2:4] - x[:, :2]).prod(axis=1)
        area_y = (y[:, 2:4] - y[:, :2]).prod(axis=1)
        ious = intersection / (area_x[:, None] + area_y[None, :] - intersection)
        # Maximum-IoU pairs fail closed if remaining boxes cannot be associated tightly.
        for _ in range(len(x)):
            row, column = np.unravel_index(np.argmax(ious), ious.shape)
            if ious[row, column] < .9:
                raise ValueError("no_one_to_one_box_match_at_iou_0_9")
            matches[row] = column
            matched_ious.append(float(ious[row, column]))
            ious[row, :] = ious[:, column] = -1
    y, v = y[matches], v[matches]
    maximum = lambda value: float(np.max(np.abs(value))) if value.size else 0.0
    differences = {"max_box_difference_px": maximum(x[:, :4] - y[:, :4]),
                   "max_corner_difference_px": maximum(u[:, :, :2] - v[:, :, :2]),
                   "max_detection_confidence_difference": maximum(x[:, 4] - y[:, 4]),
                   "max_keypoint_confidence_difference": maximum(u[:, :, 2] - v[:, :, 2])}
    passed = (differences["max_box_difference_px"] < 2.0 and
              differences["max_corner_difference_px"] < 2.0 and
              differences["max_detection_confidence_difference"] < .01 and
              differences["max_keypoint_confidence_difference"] < .01)
    return {**differences, "detections": len(x), "matched_onnx_indices": matches.tolist(),
            "min_matched_box_iou": min(matched_ious) if matched_ious else None,
            "empty_agreement_only": not len(x), "parity_passed": passed,
            "thresholds": {"coordinate_difference_px_exclusive": 2.0,
                           "confidence_difference_exclusive": .01, "minimum_box_iou": .9}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate", "export", "compare"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/training/pose"))
    parser.add_argument("--scale", choices=("n", "s"), default="n")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--model-yaml", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "0"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, choices=(320, 480, 640), default=480)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.action in ("train", "evaluate"):
        if not args.data:
            parser.error(args.action + " requires --data <dataset_dir>/pose.yaml")
        if args.action == "train":
            result = train(args.data, args.checkpoint, args.output, epochs=args.epochs, batch=args.batch,
                           imgsz=args.imgsz, device=args.device, resume=args.resume,
                           model_yaml=args.model_yaml, scale=args.scale)
        else:
            result = evaluate(args.data, args.checkpoint, args.output, split=args.split,
                              batch=args.batch, imgsz=args.imgsz, device=args.device)
        print(json.dumps(result, indent=2))
        return
    checkpoint = _local_checkpoint(args.checkpoint)
    device = _device(args.device)
    YOLO = _yolo()
    model = YOLO(str(checkpoint), task="pose")
    _schema(model)
    if args.action == "export":
        output = Path(model.export(format="onnx", imgsz=args.imgsz, device=device,
                                   dynamic=False, simplify=False))
        _write_json(output.with_suffix(".manifest.json"), {
            "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
            "onnx_sha256": _sha256(output), "imgsz": args.imgsz, "validated_after_export": False})
        return
    if not args.image or not args.image.is_file():
        parser.error("compare requires a local --image")
    exported = checkpoint.with_suffix(".onnx")
    if not exported.is_file():
        parser.error("compare requires exported ONNX alongside checkpoint")
    options = dict(imgsz=args.imgsz, device=device, rect=False, conf=.25, iou=.7,
                   max_det=12, verbose=False)
    before = model.predict(str(args.image), **options)[0]
    after = YOLO(str(exported), task="pose").predict(str(args.image), **options)[0]
    try:
        comparison = _compare_pose_results(before, after)
    except ValueError as exc:
        raise SystemExit("Cannot establish pose parity: " + str(exc)) from exc
    print(json.dumps({"image": str(args.image), "imgsz": args.imgsz, **comparison}, allow_nan=False))
    if not comparison["parity_passed"]:
        raise SystemExit("Export parity exceeds box/corner/confidence thresholds; inspect matches before deployment")


if __name__ == "__main__":
    main()
