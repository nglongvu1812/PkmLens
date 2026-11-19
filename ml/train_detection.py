"""Real YOLO26 training entry point; metrics and activation evidence are persisted."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import shutil


def preserve_resume_checkpoint(trainer):
    # Ultralytics final_eval strips optimizer state from last.pt, even after an early stop.
    source = Path(trainer.last)
    target = source.parent / 'resume.pt'
    temp = target.with_suffix('.tmp')
    shutil.copy2(source, temp)
    temp.replace(target)


def train(dataset, output, model="n", epochs=25, batch=4, imgsz=480, device="auto", resume=None):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    output, dataset = Path(output).resolve(), Path(dataset).resolve()
    if resume:
        previous = json.loads((Path(resume).resolve().parents[1] / 'config.json').read_text(encoding='utf-8'))
        if Path(previous['dataset']).resolve() != dataset:
            raise ValueError('resume_dataset_must_match_original')
        model, epochs, batch, imgsz = (previous.get(key, value) for key, value in
                                      [('model', model), ('epochs', epochs), ('batch', batch), ('imgsz', imgsz)])
    if not 1 <= int(epochs) <= 300 or not 1 <= int(batch) <= 16 or int(imgsz) not in (320, 480, 640):
        raise ValueError('invalid_training_configuration')
    if not (dataset / "dataset.yaml").is_file():
        raise ValueError("dataset_yaml_missing")
    metadata = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        if not torch.cuda.is_available():
            raise ValueError("cuda_unavailable_choose_cpu")
        free, total = torch.cuda.mem_get_info()
        if free < 2 * 1024 ** 3:
            raise ValueError("insufficient_free_vram")
        batch = min(batch, 4 if free < 5 * 1024 ** 3 else 8)
    output.mkdir(parents=True, exist_ok=True)
    config = {"model": model, "epochs": epochs, "batch": batch, "imgsz": imgsz, "device": device,
              "dataset": str(dataset), "dataset_kind": metadata["kind"], "started_at": time.time()}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    local_base = Path(os.getenv('DATA_DIR', 'data')) / 'models' / f'yolo26{model}.pt'
    detector = YOLO(str(resume) if resume else str(local_base) if local_base.exists() else f"yolo26{model}.pt")

    def epoch_end(trainer):
        status = {"status": "training", "epoch": trainer.epoch + 1, "epochs": trainer.epochs,
                  "metrics": {str(k): float(v) for k, v in trainer.metrics.items()}, "updated_at": time.time()}
        (output / "status.json").write_text(json.dumps(status), encoding="utf-8")
        if (output / "STOP").exists():
            trainer.stop = True

    detector.add_callback("on_fit_epoch_end", epoch_end)
    detector.add_callback('on_model_save', preserve_resume_checkpoint)
    use_amp = device != 'cpu' and '1660' not in torch.cuda.get_device_name(0)
    try:
        detector.train(data=str(dataset / "dataset.yaml"), epochs=epochs, batch=batch, imgsz=imgsz,
                       device=device, workers=0, amp=use_amp, project=str(output.parent), name=output.name,
                       exist_ok=True, resume=bool(resume), seed=26, fliplr=0, flipud=0, degrees=0,
                       cache=False, plots=True, patience=8, save=True, save_period=5, mosaic=.3)
        checkpoint = output / "weights/best.pt"
        measured = YOLO(str(checkpoint)).val(data=str(dataset / "dataset.yaml"), split="val", device=device,
                                             imgsz=imgsz, batch=batch, workers=0, plots=False)
        metrics = {str(k): float(v) for k, v in measured.results_dict.items()}
        passed = (metrics.get("metrics/mAP50(B)", 0) >= .75 and metrics.get("metrics/precision(B)", 0) >= .7
                  and metrics.get("metrics/recall(B)", 0) >= .7)
        result = {"status": "stopped" if (output / "STOP").exists() else "completed", "metrics": metrics,
                   "validation_passed": passed, "real_photo_validated": metadata.get("real_photo_validated") is True,
                  "checkpoint": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "dataset_sha256": metadata["sha256"], "dataset_kind": metadata["kind"],
                  "model": f"yolo26{model}", "device": device, "finished_at": time.time(), "classes": ["pokemon_card"]}
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output / "status.json").write_text(json.dumps(result), encoding="utf-8")
        return result
    except Exception as exc:
        (output / "status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}), encoding="utf-8")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--model", choices=["n", "s", "m"], default="n")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    print(json.dumps(train(a.dataset, a.output, a.model, a.epochs, a.batch, a.imgsz, a.device, a.resume), indent=2))
