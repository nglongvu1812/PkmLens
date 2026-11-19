"""Verified card checkpoints; YOLO weights are shared, trackers are session-local."""
import gc
import hashlib
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
from .recognition_config import DEFAULT_POLICY

log = logging.getLogger(__name__)
DATA = Path(os.getenv("DATA_DIR", "data")).resolve()
model_lock = threading.Lock()
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pkmlens-gpu")
_gpu_tid = None
_model = None
_loaded = None
_pose_model = None
_pose_loaded = None


def run_on_gpu(fn, *args, **kwargs):
    """Pin CUDA inference to one persistent OS thread to avoid Windows per-thread cuDNN init stalls."""
    if threading.get_ident() == _gpu_tid:
        return fn(*args, **kwargs)
    def _wrapped():
        global _gpu_tid
        _gpu_tid = threading.get_ident()
        return fn(*args, **kwargs)
    return _gpu_executor.submit(_wrapped).result()


def registry():
    path = DATA / "models/detectors.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"active": None, "models": []}


def resolve_model(model_id=None):
    data = registry()
    selected = model_id or data["active"]
    info = next((m for m in data["models"] if m["id"] == selected), None)
    if not info or not info.get("validation_passed") or info.get("classes") != ["pokemon_card"]:
        raise ValueError("card_detector_not_activated")
    path = Path(info["checkpoint"]).resolve()
    if not path.is_relative_to(DATA) or not path.is_file():
        raise ValueError("detector_checkpoint_missing")
    return info, path


def pose_registry():
    path = DATA / "models/pose-detectors.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"active": None, "models": []}


def resolve_pose_model(model_id=None, verify_checksum=True):
    data = pose_registry()
    selected = model_id or data.get("active")
    info = next((m for m in data.get("models", []) if m.get("id") == selected), None)
    if (not selected or not info or info.get("validation_passed") is not True or info.get("task") != "pose"
            or info.get("classes") != ["pokemon_card"]
            or info.get("keypoints") != ["TL", "TR", "BR", "BL"]):
        raise ValueError("pose_detector_not_activated")
    path = Path(info.get("checkpoint", "")).resolve()
    if not path.is_relative_to(DATA) or not path.is_file():
        raise ValueError("pose_detector_checkpoint_missing")
    if verify_checksum and hashlib.sha256(path.read_bytes()).hexdigest() != info.get("sha256"):
        raise ValueError("pose_detector_checksum_mismatch")
    return info, path


def public_model_metadata(info):
    """Only fixed schema and typed scalar fields may cross the browser boundary."""
    model_id = info.get("id")
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", model_id):
        raise ValueError("invalid_detector_id")
    value = {"id": model_id, "classes": ["pokemon_card"],
             "validation_passed": info.get("validation_passed") is True,
             "real_photo_validated": info.get("real_photo_validated") is True}
    label = info.get("model")
    value["model"] = label if isinstance(label, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", label) else model_id
    if info.get("task") == "pose":
        value.update(task="pose", keypoints=["TL", "TR", "BR", "BL"])
    return value


def verified_pose_metadata():
    """Readiness is lazy: activation verifies architecture; boundaries verify metadata/hash."""
    try:
        info, _ = resolve_pose_model()
        model = public_model_metadata(info)
        return {"pose_ready": True, "pose_active": model["id"], "pose_models": [model]}
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return {"pose_ready": False, "pose_active": None, "pose_models": []}


def _verify_pose_schema(model):
    arch = getattr(model, "model", model)
    names = getattr(model, "names", None)
    names = list(names.values()) if isinstance(names, dict) else names
    task = getattr(arch, "task", None)
    if task is None:
        task = getattr(arch, "args", {}).get("task")
    if (task != "pose" or getattr(model, "task", None) != "pose" or names != ["pokemon_card"]
            or getattr(arch, "kpt_shape", None) not in ([4, 3], (4, 3))):
        raise ValueError("not_a_four_corner_pokemon_pose_model")
    return {"task": "pose", "classes": ["pokemon_card"], "keypoints": ["TL", "TR", "BR", "BL"]}


def verify_pose_checkpoint(path):
    """Inspect a local checkpoint without changing either shared inference cache."""
    from ultralytics import YOLO
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise ValueError("pose_detector_checkpoint_missing")
    try:
        return _verify_pose_schema(YOLO(str(path), task="pose"))
    except Exception as exc:
        raise ValueError("not_a_four_corner_pokemon_pose_model") from exc


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def detect_pose(image, model_id=None, confidence=.45, size=480):
    global _pose_model, _pose_loaded
    import torch
    from ultralytics import YOLO
    info, path = resolve_pose_model(model_id, verify_checksum=False)
    device = os.getenv("DETECTOR_DEVICE", "auto")
    device = "0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    stat = path.stat()
    signature = (info["id"], str(path), info.get("sha256"), device,
                 stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    def _run():
        global _pose_model, _pose_loaded
        with model_lock:
            if _pose_loaded != signature:
                # Hash only on load or a changed file boundary, never on an unchanged frame.
                if hashlib.sha256(path.read_bytes()).hexdigest() != info.get("sha256"):
                    raise ValueError("pose_detector_checksum_mismatch")
                model = YOLO(str(path), task="pose")
                _verify_pose_schema(model)
                _pose_model, _pose_loaded = model, signature
                log.info("pose_detector_loaded id=%s device=%s", info["id"], device)
            return _pose_model.predict(image, conf=max(.1, confidence * .5), imgsz=size,
                                       device=device, half=device != "cpu", max_det=24, verbose=False)[0]
    result = run_on_gpu(_run) if device != "cpu" else _run()
    boxes = result.boxes
    xyxy = _numpy(boxes.xyxy).reshape(-1, 4)
    rows = np.concatenate((xyxy, _numpy(boxes.conf).reshape(-1, 1),
                           _numpy(boxes.cls).reshape(-1, 1)), axis=1)
    # Keep original row indices: ByteTrack's final column indexes this full set.
    corners = np.full((len(rows), 4, 2), np.nan, dtype=np.float32)
    keypoints = getattr(result, "keypoints", None)
    if keypoints is not None:
        points = _numpy(keypoints.xy)
        confidence_values = getattr(keypoints, "conf", None)
        if points.shape == corners.shape and confidence_values is not None:
            scores = _numpy(confidence_values)
            if scores.shape == (len(rows), 4):
                valid = (np.isfinite(scores).all(axis=1) & (scores <= 1).all(axis=1)
                         & (scores >= DEFAULT_POLICY.pose_min_keypoint_confidence).all(axis=1))
                w, h = float(image.width), float(image.height)
                in_bounds = (np.isfinite(points).all(axis=(1, 2))
                             & (points[..., 0] >= -w * .02).all(axis=1) & (points[..., 0] <= w * 1.02).all(axis=1)
                             & (points[..., 1] >= -h * .02).all(axis=1) & (points[..., 1] <= h * 1.02).all(axis=1))
                clamped = points.copy()
                if w > 1 and h > 1:
                    clamped[..., 0] = np.clip(clamped[..., 0], 0.0, w - 1e-3)
                    clamped[..., 1] = np.clip(clamped[..., 1], 0.0, h - 1e-3)
                corners[valid] = np.where(in_bounds[valid, None, None], clamped[valid], points[valid])
    invalid = (~np.isfinite(rows).all(axis=1) | (rows[:, 2] <= rows[:, 0])
               | (rows[:, 3] <= rows[:, 1]) | (rows[:, 4] < 0) | (rows[:, 4] > 1) | (rows[:, 5] != 0))
    rows[invalid] = 0
    corners[invalid] = np.nan
    return rows, corners, info


def validate_pose_corners(corners, width, height, policy=DEFAULT_POLICY):
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("invalid_pose_corners_or_confidence")
    if (points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any():
        raise ValueError("corners_outside_image")
    if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < width * height * policy.pose_min_area_ratio:
        raise ValueError("corners_not_convex_or_too_small")
    if cv2.contourArea(points, oriented=True) <= 0:
        raise ValueError("corner_order_must_be_TL_TR_BR_BL")
    if np.min(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)) < policy.pose_min_edge_pixels:
        raise ValueError("pose_edge_too_small")
    return points


def rectify_corners(image, corners, policy=DEFAULT_POLICY):
    points = validate_pose_corners(corners, image.width, image.height, policy)
    target = np.float32([[0, 0], [503, 0], [503, 703], [0, 703]])
    flat = cv2.warpPerspective(np.asarray(image), cv2.getPerspectiveTransform(points, target), (504, 704))
    return Image.fromarray(flat), points.tolist()


def normalize_corners(corners, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("invalid_image_dimensions")
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("invalid_corners")
    return (points / np.array([width, height], dtype=np.float32)).tolist()


def detect(image, model_id=None, confidence=.45, size=480):
    global _model, _loaded
    import torch
    from ultralytics import YOLO
    info, path = resolve_model(model_id)
    device = os.getenv("DETECTOR_DEVICE", "auto")
    device = "0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    signature = (info["sha256"], device)
    def _run():
        global _model, _loaded
        with model_lock:
            if _loaded != signature:
                if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                    raise ValueError("detector_checksum_mismatch")
                model = YOLO(str(path))
                if list(model.names.values()) != ["pokemon_card"]:
                    raise ValueError("not_a_card_trained_model")
                _model, _loaded = model, signature
                log.info("detector_loaded id=%s device=%s", info["id"], device)
            result = _model.predict(image, conf=max(.1, confidence * .5), imgsz=size, device=device,
                                    half=device != "cpu", max_det=24, verbose=False)[0]
            return result.boxes.cpu().numpy(), info
    return run_on_gpu(_run) if device != "cpu" else _run()


def tracker_input(rows, image_shape):
    from ultralytics.engine.results import Boxes
    boxes = rows.cpu().numpy() if isinstance(rows, Boxes) else None
    values = boxes.data if boxes is not None else np.asarray(rows, dtype=np.float32)
    if values.size == 0:
        values = values.reshape(0, 6)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError("invalid_tracker_detections")
    return boxes if boxes is not None else Boxes(values, image_shape)


def new_tracker(confidence=.45):
    from ultralytics.trackers.byte_tracker import BYTETracker
    return BYTETracker(SimpleNamespace(track_high_thresh=confidence, track_low_thresh=.1,
                       new_track_thresh=confidence, track_buffer=12, match_thresh=.7, fuse_score=True))


def unload():
    global _model, _loaded, _pose_model, _pose_loaded, _last_pose_frame
    import torch
    with model_lock:
        _model = _loaded = _pose_model = _pose_loaded = _last_pose_frame = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_last_pose_frame = None


def _pose_rectify_crop(image, box):
    global _last_pose_frame
    try:
        cached = _last_pose_frame
        if cached is not None and cached[0] is image:
            rows, corners = cached[1], cached[2]
        else:
            rows, corners, _ = detect_pose(image, confidence=.25, size=480)
            _last_pose_frame = (image, rows, corners)
        if not len(rows):
            return None
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) * .5, (y1 + y2) * .5
        diag = max(32.0, float(np.hypot(x2 - x1, y2 - y1)))
        best_idx, best_dist = None, (diag * .65) ** 2
        for idx, row in enumerate(rows):
            if not np.isfinite(corners[idx]).all():
                continue
            rcx, rcy = (row[0] + row[2]) * .5, (row[1] + row[3]) * .5
            dist = (rcx - cx) ** 2 + (rcy - cy) ** 2
            if dist < best_dist:
                best_idx, best_dist = idx, dist
        if best_idx is None:
            return None
        flat, _ = rectify_corners(image, corners[best_idx])
        return flat
    except Exception:
        return None


def rectify(image, box):
    """Contour correction inside a detected box; uncertain contours use a labeled bbox fallback."""
    x1, y1, x2, y2 = [int(v) for v in box]
    width, height = image.size
    x1, y1, x2, y2 = max(0,x1-4), max(0,y1-4), min(width,x2+4), min(height,y2+4)
    if x2-x1 < 32 or y2-y1 < 32:
        raise ValueError("card_too_small")
    crop = image.crop((x1,y1,x2,y2))
    gray = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
    tilted = min(crop.width, crop.height) / max(crop.width, crop.height) > .74
    has_texture = float(gray.std()) >= 5.0
    if tilted and has_texture:
        pose_flat = _pose_rectify_crop(image, (x1, y1, x2, y2))
        if pose_flat is not None:
            return pose_flat, "contour"
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),40,120)
    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:5]:
        quad = cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True).reshape(-1,2)
        if len(quad)!=4 or not cv2.isContourConvex(quad) or cv2.contourArea(quad)<gray.size*.4:
            continue
        center=quad.mean(0)
        quad=quad[np.argsort(np.arctan2(quad[:,1]-center[1],quad[:,0]-center[0]))].astype(np.float32)
        quad=np.roll(quad,-int(np.argmin(quad.sum(1))),axis=0)
        if np.linalg.norm(quad[1]-quad[0]) > np.linalg.norm(quad[2]-quad[1]):
            quad=np.roll(quad,-1,axis=0)
        target=np.float32([[0,0],[503,0],[503,703],[0,703]])
        flat=cv2.warpPerspective(np.array(crop),cv2.getPerspectiveTransform(quad,target),(504,704))
        return Image.fromarray(flat), "contour"
    if crop.width > crop.height:
        crop=crop.rotate(90,expand=True)
    return crop.resize((504,704)), "bbox_fallback"


def quality(image, policy=DEFAULT_POLICY):
    gray=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2GRAY)
    sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    brightness=float(gray.mean())
    glare=float(np.mean(gray>250))
    reasons=[]
    if sharp<policy.min_sharpness: reasons.append("blurred")
    if brightness<policy.min_brightness: reasons.append("too_dark")
    if glare>policy.max_glare: reasons.append("glare")
    return {"sharpness":round(sharp,2),"brightness":round(brightness,2),"glare":round(glare,3),
            "usable":not reasons,"reasons":reasons}
