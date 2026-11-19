import json
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import JSONResponse
from .imaging import decode, MAX_BYTES
from .pipeline import identify, active_cardcorpus_gallery, cardcorpus_bundle_status

app = FastAPI(title="PkmLens AI", version="0.1.0")

from .live import router as live_router
app.include_router(live_router)

from .training import router as training_router
app.include_router(training_router)

@app.get('/v1/detectors')
def detectors(x_cardlens_key: str = Header(default='')):
    import secrets
    key=os.getenv('AI_INTERNAL_KEY','')
    if not key or not secrets.compare_digest(x_cardlens_key,key):
        raise HTTPException(403,'internal_service_only')
    from .detection import registry, verified_pose_metadata, public_model_metadata
    import torch
    data=registry();pose=verified_pose_metadata();gallery=cardcorpus_bundle_status()
    return {'available':bool(data['active']),'active':data['active'],
            'models':[public_model_metadata(m) for m in data['models']],
            'pose_active':pose['pose_active'],'pose_models':pose['pose_models'],
            'gallery':gallery,
            'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

@app.get("/healthz")
def health():
    root = Path(os.getenv("DATA_DIR", "data")) / "models"
    files=(root/"registry.json",root/"gallery.faiss",root/"gallery-phash.npy",root/"dinov2-small"/"model.safetensors",root/"dinov2-small"/"checksums.json")
    from .detection import verified_pose_metadata
    pose_ready = verified_pose_metadata()["pose_ready"]
    try:
        metadata = json.loads((root / "registry.json").read_text(encoding="utf-8"))
        checkpoint_kind = metadata.get("checkpoint_kind", "unavailable") if isinstance(metadata, dict) else "unavailable"
    except (OSError, ValueError):
        checkpoint_kind = "unavailable"
    gallery = cardcorpus_bundle_status()
    identification_ready = gallery["available"] and (gallery["active"] == "cardcorpus" or all(path.is_file() for path in files))
    return {"status": "ok" if identification_ready else "unavailable", "pose_ready": pose_ready,
            "identification_ready": identification_ready,
            "checkpoint_kind": "external_pretrained_matcher" if gallery["active"] == "cardcorpus" else checkpoint_kind,
            "gallery": gallery, "supported_modes": ["cropped", "manual_corners"]}

@app.get("/readyz")
def ready():
    value = health()
    return JSONResponse(value, status_code=200 if value["identification_ready"] else 503)

@app.post("/v1/identify")
def scan(image: UploadFile = File(), mode: str = Form("cropped"), corners: str | None = Form(None)):
    raw = image.file.read(MAX_BYTES+1)
    try:
        decoded = decode(raw)
        if os.getenv("PKMLENS_WARMUP_ON_START") == "1":
            from .detection import detect, detect_pose
            try:
                detect(decoded)
                detect_pose(decoded)
            except Exception:
                pass
        return identify(decoded, mode, json.loads(corners) if corners else None)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "gallery_or_embedding_weight_missing") from exc

