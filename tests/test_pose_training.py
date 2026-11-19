"""Pose lifecycle checks; the trainer boundary is faked, never the dataset validator."""
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import yaml

from ml import train_pose


@pytest.fixture
def dataset(tmp_path):
    from PIL import Image

    root = tmp_path / "dataset"
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        name = hashlib.sha256(split.encode()).hexdigest()
        images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        image = images / (name + ".jpg")
        Image.new("RGB", (100, 160), (30 + index * 60, 70, 110)).save(image)
        (labels / (name + ".txt")).write_text(
            "0 .5 .5 .8 .875 .1 .0625 2 .9 .0625 2 .9 .9375 2 .1 .9375 2\n",
            encoding="utf-8",
        )
        rows.append({"sample_id": split, "split": split, "export_id": name,
                     "source_kind": "synthetic_catalog_scene",
                     "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                     "objects": [{"group": "art-" + split,
                                  "corners": [[10, 10], [90, 10], [90, 150], [10, 150]]}]})
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    (root / "pose.yaml").write_text(yaml.safe_dump({
        "path": str(root), "train": "images/train", "val": "images/val", "test": "images/test",
        "kpt_shape": [4, 3], "flip_idx": [1, 0, 3, 2], "names": {0: "pokemon_card"},
    }), encoding="utf-8")
    return root


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import ultralytics
    import ultralytics.data.dataset

    calls = {"instances": [], "train": [], "val": [], "saved_resume": []}
    package_file = ultralytics.__file__

    class Model:
        def __init__(self, path, task=None):
            self.path = Path(path)
            self.task = "pose"
            self.ckpt = None
            if self.path.suffix == ".yaml":
                config = yaml.safe_load(self.path.read_text(encoding="utf-8"))
                # Real Ultralytics assigns numeric default names to a new YAML model.
                names = {0: "0"}
            else:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                self.ckpt = saved
                config = saved["yaml"]
                names = {int(key): value for key, value in saved.get("names", {0: "pokemon_card"}).items()}
                self.task = saved.get("task", "pose")
            self.model = types.SimpleNamespace(yaml=config, kpt_shape=config.get("kpt_shape"), names=names)
            self.callbacks = {}
            calls["instances"].append(self)

        @property
        def names(self):
            return self.model.names

        def load(self, path):
            calls["initializer"] = str(path)
            self.ckpt = {"epoch": -1, "optimizer": None}
            return self

        def save(self, *args, **kwargs):
            pytest.fail("A warm start must not save or reload a fake initialization checkpoint")

        def add_callback(self, event, function):
            self.callbacks.setdefault(event, []).append(function)

        def train(self, **kwargs):
            calls["train"].append(kwargs)
            run = Path(kwargs["project"]) / kwargs["name"]
            assert run == Path(kwargs.get("save_dir", run))
            if calls.get("on_start"):
                calls["on_start"](run)
            weights = run / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            first = self.ckpt["epoch"] + 1 if kwargs["resume"] else 0
            trainer = types.SimpleNamespace(epoch=first, epochs=kwargs["epochs"], metrics={},
                                            stop=False, last=weights / "last.pt", save_dir=run)
            for epoch in range(first, kwargs["epochs"]):
                trainer.epoch = epoch
                saved = {"epoch": epoch, "optimizer": {"state": {"0": {"step": 1}},
                                                        "param_groups": [{"params": [0]}]},
                         "yaml": self.model.yaml, "names": {0: "pokemon_card"}, "task": "pose",
                         "train_args": {**kwargs, "task": "pose", "save_dir": str(run),
                                        "model": str(self.path)}}
                for name in ("last.pt", "best.pt"):
                    (weights / name).write_text(json.dumps(saved), encoding="utf-8")
                for function in self.callbacks.get("on_model_save", []):
                    function(trainer)
                calls["saved_resume"].append(json.loads((weights / "resume.pt").read_text()))
                for function in self.callbacks.get("on_fit_epoch_end", []):
                    function(trainer)
                if trainer.stop:
                    break
            # Reproduce final_eval: strip last/best, then emit a spurious extra epoch callback.
            for name in ("last.pt", "best.pt"):
                stripped = json.loads((weights / name).read_text())
                stripped.update(epoch=-1, optimizer=None)
                (weights / name).write_text(json.dumps(stripped), encoding="utf-8")
            trainer.epoch += 1
            for function in self.callbacks.get("on_fit_epoch_end", []):
                function(trainer)
            self.trainer = trainer

        def val(self, **kwargs):
            calls["val"].append(kwargs)
            return types.SimpleNamespace(results_dict={"metrics/precision(P)": .9,
                "metrics/recall(P)": .85, "metrics/mAP50(P)": .88, "metrics/mAP50-95(P)": .7})

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Model, __file__=package_file))
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"trusted local initializer (load boundary faked)")
    calls["checkpoint"] = initializer
    return calls


def run_training(dataset, runtime, output, **kwargs):
    return train_pose.train(dataset / "pose.yaml", runtime["checkpoint"], output,
                            epochs=3, batch=1, imgsz=320, device="cpu", **kwargs)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_fresh_training_builds_run_specific_scale_and_uses_direct_local_warm_start(dataset, runtime, tmp_path):
    output = tmp_path / "run"
    result = run_training(dataset, runtime, output, scale="s")

    architecture = output / "yolo26s-card-pose.yaml"
    config = yaml.safe_load(architecture.read_text())
    assert config["scale"] == "s" and config["kpt_shape"] == [4, 3] and config["nc"] == 1
    assert config["head"][-1][2] == "Pose26"
    assert runtime["instances"][0].path == architecture
    assert runtime["initializer"] == str(runtime["checkpoint"])
    assert runtime["train"][0]["amp"] is False
    assert runtime["train"][0]["exist_ok"] is True
    assert runtime["train"][0]["fliplr"] == runtime["train"][0]["flipud"] == 0
    saved = read_json(output / "pose-config.json")
    assert saved["dataset"] == str(dataset.resolve()) and saved["checkpoint"] == str(runtime["checkpoint"])
    assert saved["scale"] == "s" and saved["task"] == "pose"
    assert result["status"] == "completed" and result["epoch"] == 3
    assert result["validation_passed"] and result["real_photo_validated"] is False
    assert result["source_kinds"] == ["synthetic_catalog_scene"]
    assert result["validation_scope"] == "synthetic_validation_only"
    assert result["sha256"] == hashlib.sha256(Path(result["checkpoint"]).read_bytes()).hexdigest()
    assert result["metrics"]["metrics/mAP50(P)"] == .88
    assert runtime["val"][0]["imgsz"] == 320
    assert read_json(output / "pose-status.json")["epoch"] == 3
    assert read_json(output / "weights/resume.pt")["epoch"] == 2
    assert read_json(output / "weights/last.pt")["optimizer"] is None
    assert not (output.parent / "pose-warm-start.pt").exists()


def test_stop_created_during_validation_is_honored_after_first_epoch(dataset, runtime, tmp_path, monkeypatch):
    output = tmp_path / "run"
    validator = train_pose.validate_pose_dataset

    def request_stop(root):
        assert read_json(output / "pose-status.json")["status"] == "starting"
        (output / "STOP").touch()
        return validator(root)

    monkeypatch.setattr(train_pose, "validate_pose_dataset", request_stop)
    result = run_training(dataset, runtime, output)

    assert result["status"] == "stopped" and result["epoch"] == 1
    assert read_json(output / "weights/resume.pt")["optimizer"]["state"]
    assert len(runtime["saved_resume"]) == 1


def stopped_run(dataset, runtime, output):
    def stop(trainer):
        (output / "STOP").touch()
        trainer.stop = True
    run_training(dataset, runtime, output, callback=stop)


def test_resume_preserves_initializer_config_and_invalidates_old_validation_gate(dataset, runtime, tmp_path):
    output = tmp_path / "run"
    stopped_run(dataset, runtime, output)
    config = (output / "pose-config.json").read_bytes()
    previous = read_json(output / "pose-result.json")
    previous.update(status="completed", validation_passed=True)
    (output / "pose-result.json").write_text(json.dumps(previous))

    def check_gate(run):
        state = read_json(run / "pose-result.json")
        assert state["status"] == "starting" and state["validation_passed"] is False
        assert not (run / "STOP").exists()

    runtime["on_start"] = check_gate
    result = train_pose.train(dataset / "pose.yaml", output / "weights/resume.pt", output,
                              epochs=3, batch=1, imgsz=320, device="cpu", resume=True)

    assert result["status"] == "completed" and result["epoch"] == 3
    assert runtime["train"][-1]["resume"] is True
    assert (output / "pose-config.json").read_bytes() == config
    assert result["initial_checkpoint"] == str(runtime["checkpoint"])


@pytest.mark.parametrize("change,error", [
    ({"epoch": -1}, "resumable"),
    ({"optimizer": None}, "resumable"),
    ({"optimizer": {"state": {}, "param_groups": []}}, "resumable"),
    ({"task": "detect"}, "four_corner_pose"),
    ({"names": {0: "person"}}, "four_corner_pose"),
    ({"yaml.kpt_shape": [17, 3]}, "four_corner_pose"),
    ({"train_args.save_dir": "some-other-run"}, "same_run"),
    ({"train_args.data": "other/pose.yaml"}, "dataset"),
    ({"train_args.batch": 2}, "configuration"),
])
def test_resume_rejects_stripped_wrong_schema_or_foreign_state(dataset, runtime, tmp_path, change, error):
    output = tmp_path / "run"
    stopped_run(dataset, runtime, output)
    checkpoint = output / "weights/resume.pt"
    payload = read_json(checkpoint)
    for key, value in change.items():
        if "." in key:
            parent, child = key.split(".")
            payload[parent][child] = value
        else:
            payload[key] = value
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=error):
        train_pose.train(dataset / "pose.yaml", checkpoint, output, epochs=3, batch=1, imgsz=320,
                         device="cpu", resume=True)
    assert len(runtime["train"]) == 1


def test_resume_rejects_changed_dataset_and_fresh_training_cannot_reuse_run(dataset, runtime, tmp_path):
    output = tmp_path / "run"
    stopped_run(dataset, runtime, output)
    saved = (output / "pose-config.json").read_bytes()
    with pytest.raises(ValueError, match="exists"):
        run_training(dataset, runtime, output)
    label = next((dataset / "labels/train").glob("*.txt"))
    label.write_text(label.read_text() + "\n")
    with pytest.raises(ValueError, match="dataset"):
        train_pose.train(dataset / "pose.yaml", output / "weights/resume.pt", output,
                         epochs=3, batch=1, imgsz=320, device="cpu", resume=True)
    assert (output / "pose-config.json").read_bytes() == saved
    assert len(runtime["train"]) == 1


def test_validation_failure_records_failed_status_without_loading_weights(dataset, runtime, tmp_path):
    output = tmp_path / "run"
    next((dataset / "labels/train").glob("*.txt")).write_text("invalid label\n")
    with pytest.raises(ValueError, match="pose_dataset_invalid"):
        run_training(dataset, runtime, output)
    assert read_json(output / "pose-status.json")["status"] == "failed"
    assert read_json(output / "pose-result.json")["validation_passed"] is False
    assert not runtime["instances"]


def test_evaluate_uses_selected_split_size_and_never_claims_camera_validation(dataset, runtime, tmp_path):
    run = tmp_path / "run"
    run_training(dataset, runtime, run)
    report = train_pose.evaluate(dataset / "pose.yaml", run / "weights/best.pt", tmp_path / "evaluation",
                                 split="test", batch=1, imgsz=640, device="cpu")
    assert runtime["val"][-1]["split"] == "test" and runtime["val"][-1]["imgsz"] == 640
    assert report["split"] == "test" and report["real_photo_validated"] is False
    assert read_json(tmp_path / "evaluation/pose-evaluation.json")["sha256"] == report["sha256"]


def test_cli_passes_scale_batch_size_and_evaluation_split(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(train_pose, "train", lambda *args, **kwargs: calls.append((args, kwargs)) or {})
    monkeypatch.setattr(sys, "argv", ["ml.train_pose", "train", "--data", "dataset/pose.yaml",
        "--checkpoint", "init.pt", "--output", str(tmp_path), "--scale", "s", "--epochs", "2",
        "--batch", "3", "--imgsz", "640", "--device", "auto"])
    train_pose.main()
    assert calls[0][1]["scale"] == "s" and calls[0][1]["batch"] == 3
    assert calls[0][1]["imgsz"] == 640 and calls[0][1]["device"] == "auto"
    monkeypatch.setattr(train_pose, "evaluate", lambda *args, **kwargs: calls.append((args, kwargs)) or {})
    monkeypatch.setattr(sys, "argv", ["ml.train_pose", "evaluate", "--data", "dataset/pose.yaml",
        "--checkpoint", "best.pt", "--output", str(tmp_path), "--split", "test", "--imgsz", "320"])
    train_pose.main()
    assert calls[-1][1]["split"] == "test" and calls[-1][1]["imgsz"] == 320


def test_installed_yolo26_architecture_preserves_both_supported_scales(tmp_path):
    from ultralytics import YOLO

    for scale in ("n", "s"):
        target = tmp_path / ("yolo26" + scale + "-card-pose.yaml")
        path = train_pose.create_pose_model_yaml(target, scale)
        model = YOLO(str(path), task="pose")
        assert model.model.kpt_shape == [4, 3] and model.model.yaml["nc"] == 1
        assert model.model.yaml["scale"] == scale


def test_loader_does_not_write_or_trust_label_caches(dataset):
    from ultralytics.data.dataset import YOLODataset
    from ml.prepare_pose import validate_pose_dataset

    assert validate_pose_dataset(dataset)["valid"]
    with train_pose._uncached_labels():
        loaded = YOLODataset(str(dataset / "images/train"), imgsz=320, batch_size=1,
                             data=yaml.safe_load((dataset / "pose.yaml").read_text()), task="pose", cache=False)
    assert len(loaded) == 1
    assert loaded.labels[0]["keypoints"].shape == (1, 4, 3)
    assert not list(dataset.rglob("*.cache"))
    assert validate_pose_dataset(dataset)["valid"]


def test_legacy_model_yaml_is_copied_into_run_and_initialized_without_saving(dataset, runtime, tmp_path):
    import ultralytics

    architecture = Path(ultralytics.__file__).parent / "cfg/models/26/yolo26-pose.yaml"
    custom = yaml.safe_load(architecture.read_text(encoding="utf-8"))
    custom.update(nc=1, kpt_shape=[4, 3], names={0: "pokemon_card"})
    legacy = tmp_path / "legacy-pose.yaml"
    legacy.write_text(yaml.safe_dump(custom), encoding="utf-8")
    result = run_training(dataset, runtime, tmp_path / "run", model_yaml=legacy)
    assert result["status"] == "completed"
    assert read_json(tmp_path / "run/pose-config.json")["model_yaml_source"] == str(legacy)
    assert runtime["instances"][0].path.name == "yolo26n-card-pose.yaml"


def test_api_bootstrap_folder_can_start_and_does_not_increment_run_name(dataset, runtime, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "pose-config.json").write_text(json.dumps({"dataset": str(dataset), "checkpoint": str(runtime["checkpoint"]),
        "task": "pose", "epochs": 3, "batch": 1, "imgsz": 320, "device": "cpu"}))
    (output / "pose-status.json").write_text(json.dumps({"status": "starting", "task": "pose"}))
    (output / "pose-train.log").touch()
    result = run_training(dataset, runtime, output)
    assert Path(result["checkpoint"]).parents[1] == output
    assert runtime["train"][0]["name"] == "run"


def test_resume_new_stop_during_validation_survives_stale_stop_cleanup(dataset, runtime, tmp_path, monkeypatch):
    output = tmp_path / "run"
    stopped_run(dataset, runtime, output)
    original = train_pose.validate_pose_dataset
    def validate_and_stop(root):
        assert not (output / "STOP").exists()
        (output / "STOP").touch()
        return original(root)
    monkeypatch.setattr(train_pose, "validate_pose_dataset", validate_and_stop)
    result = train_pose.train(dataset / "pose.yaml", output / "weights/resume.pt", output,
                              epochs=3, batch=1, imgsz=320, device="cpu", resume=True)
    assert result["status"] == "stopped" and result["epoch"] == 2


@pytest.mark.parametrize("checkpoint,epochs,device", [("missing.pt", 3, "cpu"), (None, 0, "cpu"),
                                                       (None, 3, "invalid")])
def test_startup_errors_before_metrics_leave_failed_artifacts(dataset, runtime, tmp_path, checkpoint, epochs, device):
    output = tmp_path / "run"
    source = tmp_path / checkpoint if checkpoint else runtime["checkpoint"]
    with pytest.raises(ValueError):
        train_pose.train(dataset / "pose.yaml", source, output, epochs=epochs, batch=1, imgsz=320, device=device)
    assert read_json(output / "pose-status.json")["status"] == "failed"
    result = read_json(output / "pose-result.json")
    assert result["status"] == "failed" and result["validation_passed"] is False
    assert "metrics" not in result and not runtime["train"]


def test_configurable_gate_persists_thresholds_in_result_and_config(dataset, runtime, tmp_path, monkeypatch):
    monkeypatch.setenv("POSE_MIN_MAP50", ".95")
    monkeypatch.setenv("POSE_MIN_PRECISION", ".8")
    monkeypatch.setenv("POSE_MIN_RECALL", ".8")
    output = tmp_path / "run"
    result = run_training(dataset, runtime, output)
    assert result["gate_thresholds"] == {"map50": .95, "precision": .8, "recall": .8}
    assert result["validation_passed"] is False
    assert read_json(output / "pose-config.json")["gate_thresholds"] == result["gate_thresholds"]


@pytest.mark.parametrize("value", ["nan", "inf", "-0.01", "1.01", "invalid"])
def test_invalid_gate_threshold_fails_before_training(dataset, runtime, tmp_path, monkeypatch, value):
    monkeypatch.setenv("POSE_MIN_MAP50", value)
    with pytest.raises(ValueError, match="pose_gate_threshold"):
        run_training(dataset, runtime, tmp_path / "run")
    assert not runtime["train"]
    assert read_json(tmp_path / "run/pose-status.json")["status"] == "failed"


def test_resume_config_rejection_clears_stale_activation_result(dataset, runtime, tmp_path):
    output = tmp_path / "run"
    stopped_run(dataset, runtime, output)
    previous = read_json(output / "pose-result.json")
    previous.update(status="completed", validation_passed=True)
    (output / "pose-result.json").write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="configuration"):
        train_pose.train(dataset / "pose.yaml", output / "weights/resume.pt", output,
                         epochs=4, batch=1, imgsz=320, device="cpu", resume=True)
    result = read_json(output / "pose-result.json")
    assert result["status"] == "failed" and result["validation_passed"] is False
    assert len(runtime["train"]) == 1


def test_object_level_real_source_is_not_labeled_as_synthetic_validation(dataset, runtime, tmp_path):
    manifest = dataset / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    obj = rows[1]["objects"][0]
    obj.update(source_kind="real_photo", consent="research_and_evaluation", source="local-photo", session_id="val-photo-session")
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = run_training(dataset, runtime, tmp_path / "run")
    assert result["dataset_kind"] == "mixed_or_unspecified"
    assert result["validation_source_kinds"] == ["real_photo", "synthetic_catalog_scene"]
    assert result["validation_scope"] == "dataset_split_only_not_real_camera_validation"
    assert result["real_photo_validated"] is False


def test_export_and_compare_use_chosen_image_size(dataset, runtime, tmp_path, monkeypatch, capsys):
    run_training(dataset, runtime, tmp_path / "run")
    checkpoint = tmp_path / "run/weights/best.pt"
    model_class = type(runtime["instances"][0])
    calls = []
    def export(model, **kwargs):
        calls.append(kwargs)
        target = checkpoint.with_suffix(".onnx")
        target.write_bytes(b"fake ONNX export boundary")
        return str(target)
    monkeypatch.setattr(model_class, "export", export, raising=False)
    monkeypatch.setattr(sys, "argv", ["ml.train_pose", "export", "--checkpoint", str(checkpoint),
                                     "--imgsz", "640", "--device", "cpu"])
    train_pose.main()
    assert calls[-1]["imgsz"] == 640
    assert read_json(checkpoint.with_suffix(".manifest.json"))["validated_after_export"] is False
    original_init = model_class.__init__
    def initialize(model, path, task=None):
        return original_init(model, checkpoint if str(path).endswith(".onnx") else path, task=task)
    monkeypatch.setattr(model_class, "__init__", initialize)
    def predict(model, path, **kwargs):
        import torch
        calls.append(kwargs)
        return [types.SimpleNamespace(
            boxes=types.SimpleNamespace(data=torch.tensor([[0., 0., 10., 10., .9, 0.]])),
            keypoints=types.SimpleNamespace(data=torch.ones((1, 4, 3))))]
    monkeypatch.setattr(model_class, "predict", predict, raising=False)
    image = next((dataset / "images/test").glob("*.jpg"))
    monkeypatch.setattr(sys, "argv", ["ml.train_pose", "compare", "--checkpoint", str(checkpoint),
                                     "--image", str(image), "--imgsz", "320", "--device", "cpu"])
    train_pose.main()
    assert [call["imgsz"] for call in calls[-2:]] == [320, 320]
    assert json.loads(capsys.readouterr().out)["max_corner_difference_px"] == 0


@pytest.mark.parametrize("resume", [False, True])
def test_api_stop_requested_before_worker_starts_is_preserved(dataset, runtime, tmp_path, resume):
    output = tmp_path / "run"
    if resume:
        stopped_run(dataset, runtime, output)
        checkpoint = output / "weights/resume.pt"
    else:
        output.mkdir()
        checkpoint = runtime["checkpoint"]
    (output / "STOP").touch()
    (output / "pose-status.json").write_text(json.dumps({"status": "stopping", "task": "pose"}))
    result = train_pose.train(dataset / "pose.yaml", checkpoint, output, epochs=3, batch=1,
                              imgsz=320, device="cpu", resume=resume)
    assert result["status"] == "stopped" and result["epoch"] == (2 if resume else 1)


def test_nonfinite_measured_metrics_record_failure_instead_of_activation(dataset, runtime, tmp_path, monkeypatch):
    output = tmp_path / "run"
    original = runtime["instances"]
    def break_validation(run):
        model_class = type(original[0])
        monkeypatch.setattr(model_class, "val", lambda self, **kwargs:
                            types.SimpleNamespace(results_dict={"metrics/mAP50(P)": float("nan")}))
    runtime["on_start"] = break_validation
    with pytest.raises(ValueError, match="not_finite"):
        run_training(dataset, runtime, output)
    result = read_json(output / "pose-result.json")
    assert result["status"] == "failed" and result["validation_passed"] is False


@pytest.mark.parametrize("change,accepted", [
    ("reordered", True),
    ("box_coordinates", False),
    ("unmatched_boxes", False),
    ("detection_confidence", False),
    ("keypoint_confidence", False),
    ("corner_coordinates", False),
    ("different_count", False),
    ("wrong_class", False),
    ("nonfinite", False),
    ("keypoint_schema", False),
    ("invalid_confidence", False),
    ("both_empty", True),
])
def test_compare_cli_matches_boxes_and_checks_confidences(tmp_path, monkeypatch, capsys, change, accepted):
    import copy
    import numpy as np
    import torch
    from ultralytics.engine.results import Results

    checkpoint, image = tmp_path / "best.pt", tmp_path / "scene.jpg"
    checkpoint.write_bytes(b"trusted local model boundary")
    checkpoint.with_suffix(".onnx").write_bytes(b"local exported model boundary")
    image.touch()
    before = Results(np.zeros((200, 300, 3), dtype=np.uint8), str(image), {0: "pokemon_card"},
        boxes=torch.tensor([[10, 20, 90, 160, .93, 0], [150, 30, 230, 170, .87, 0]]),
        keypoints=torch.tensor([[[10, 20, .95], [90, 20, .96], [90, 160, .97], [10, 160, .98]],
                                [[150, 30, .95], [230, 30, .96], [230, 170, .97], [150, 170, .98]]]))
    after = copy.deepcopy(before)
    if change == "reordered":
        after = after[[1, 0]]
    elif change == "box_coordinates":
        after.boxes.data[:, 2] += 3
    elif change == "unmatched_boxes":
        after.boxes.data[:, :4] += 20
    elif change == "detection_confidence":
        after.boxes.data[:, 4] = .01
    elif change == "keypoint_confidence":
        after.keypoints.data[:, :, 2] = .01
    elif change == "corner_coordinates":
        after.keypoints.data[:, :, :2] += 20
    elif change == "different_count":
        after = after[[0]]
    elif change == "wrong_class":
        after.boxes.data[:, 5] = 1
    elif change == "nonfinite":
        after.boxes.data[0, 0] = float("nan")
    elif change == "keypoint_schema":
        after.keypoints.data = after.keypoints.data[:, :3, :]
    elif change == "invalid_confidence":
        after.keypoints.data[:, :, 2] = 1.01
    elif change == "both_empty":
        before, after = before[[]], after[[]]
    calls = []

    class Model:
        def __init__(self, path, task=None):
            self.result = after if str(path).endswith(".onnx") else before

        def predict(self, *args, **kwargs):
            calls.append(kwargs)
            return [self.result]

    monkeypatch.setattr(train_pose, "_yolo", lambda: Model)
    monkeypatch.setattr(train_pose, "_schema", lambda model: None)
    monkeypatch.setattr(sys, "argv", ["ml.train_pose", "compare", "--checkpoint", str(checkpoint),
                                     "--image", str(image), "--imgsz", "480", "--device", "cpu"])
    if not accepted:
        with pytest.raises(SystemExit, match="parity"):
            train_pose.main()
        return
    train_pose.main()
    result = json.loads(capsys.readouterr().out)
    assert result["parity_passed"] is True
    assert result["max_box_difference_px"] == result["max_corner_difference_px"] == 0
    assert result["max_detection_confidence_difference"] == result["max_keypoint_confidence_difference"] == 0
    assert result["matched_onnx_indices"] == ([1, 0] if change == "reordered" else [])
    assert all(call["rect"] is False and call["conf"] == .25 and call["max_det"] == 12 for call in calls)
