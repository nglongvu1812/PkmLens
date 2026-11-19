import hashlib
import json
import os
import re
import time
import zipfile
from functools import lru_cache
from pathlib import Path
import faiss
import numpy as np
import torch
from PIL import Image
from .imaging import warp, encoded, phash
from .recognition_config import DEFAULT_POLICY
from .identity import (FUSION_VERSION, PROJECTION_KIND, catalog_metadata, contained_path,
                       decision, identity_fingerprint, rank_candidates, sha256)

DATA = Path(os.getenv("DATA_DIR", "data")).resolve()
MODEL_NAME = "facebook/dinov2-small"
PREPROCESS = "rgb-resize224-bicubic-imagenet-v1"
torch.set_num_threads(int(os.getenv("AI_THREADS", "2")))

_GALLERY_UNSET = object()


def active_cardcorpus_gallery():
    """Load the checksum-verified paired gallery once; absent bundles use legacy mode."""
    mode = os.getenv("PKMLENS_GALLERY_BACKEND", "auto").strip().lower()
    if mode not in ("auto", "legacy", "cardcorpus"):
        raise ValueError("invalid_gallery_backend")
    if mode == "legacy":
        return None
    root = contained_path(DATA, "models/cardcorpus-scanner")
    if not all((root / name).is_file() for name in ("checksums.json", "mapping.json")):
        if mode == "cardcorpus":
            raise FileNotFoundError("cardcorpus_gallery_not_prepared")
        return None
    from .large_gallery import get_gallery
    device = os.getenv("DETECTOR_DEVICE", "auto")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "0":
        device = "cuda:0"
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    return get_gallery(root, device=device)


def cardcorpus_bundle_status():
    mode = os.getenv("PKMLENS_GALLERY_BACKEND", "auto").strip().lower()
    root = contained_path(DATA, "models/cardcorpus-scanner")
    prepared = all((root / name).is_file() for name in ("checksums.json", "mapping.json"))
    try:
        paired = active_cardcorpus_gallery()
    except (OSError, ValueError, RuntimeError, ImportError):
        return {"requested": mode, "active": "unavailable", "prepared": prepared,
                "large_index_loaded": False, "available": False,
                "error": "paired_gallery_invalid_or_unavailable"}
    status = {"requested": mode, "active": "cardcorpus" if paired else "legacy",
              "prepared": prepared, "large_index_loaded": paired is not None, "available": True}
    if paired is not None:
        status["coverage"] = paired.coverage
    return status


@lru_cache(maxsize=1)
def embedder():
    from transformers import Dinov2Model
    target=contained_path(DATA, "models/dinov2-small")
    checksums=json.loads(contained_path(target, "checksums.json").read_text(encoding="utf-8"))
    for name in ("config.json","model.safetensors"):
        if name not in checksums or sha256(contained_path(target, name))!=checksums[name]:
            raise ValueError("embedding_checkpoint_checksum_mismatch")
    model = Dinov2Model.from_pretrained(str(target), local_files_only=True, use_safetensors=True, weights_only=True)
    model.eval()
    model.requires_grad_(False)
    return model

def backbone_embedding(image):
    """Frozen, offline DINOv2 CLS features; no projection or backbone training."""
    array = np.array(image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    array = (array - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float().unsqueeze(0)
    with torch.inference_mode():
        vector = embedder()(pixel_values=tensor).last_hidden_state[:, 0].cpu().numpy().astype("float32")
    faiss.normalize_L2(vector)
    return vector


@lru_cache(maxsize=2)
def load_projection(path, checksum, input_dim, output_dim):
    """NPZ contains tensor arrays only: no pickle deserialization."""
    if not isinstance(input_dim, int) or not isinstance(output_dim, int) or not 1 <= output_dim <= input_dim <= 4096:
        raise ValueError("invalid_projection_dimensions")
    if sha256(path) != checksum:
        raise ValueError("projection_checksum_mismatch")
    with zipfile.ZipFile(path) as container:
        files = container.infolist()
        if len(files) != 1 or files[0].filename != "weight.npy" or files[0].file_size > input_dim * output_dim * 4 + 1024:
            raise ValueError("invalid_projection_weights")
        with container.open("weight.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("invalid_projection_weights")
            if shape != (output_dim, input_dim) or dtype != np.dtype("float32") or fortran:
                raise ValueError("invalid_projection_weights")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["weight"]:
            raise ValueError("invalid_projection_weights")
        weight = archive["weight"]
    if weight.dtype != np.float32 or weight.shape != (output_dim, input_dim) or not np.isfinite(weight).all():
        raise ValueError("invalid_projection_weights")
    return weight


def projection_for(registry, target):
    if registry.get("checkpoint_kind") != PROJECTION_KIND:
        if registry.get("projection"):
            raise ValueError("incompatible_projection_kind")
        return None
    info = registry["projection"]
    path = contained_path(target, info["file"])
    return load_projection(str(path), info["sha256"], info["input_dim"], info["output_dim"])


@lru_cache(maxsize=1)
def active_projection():
    target = contained_path(DATA, "models")
    path = contained_path(target, "registry.json")
    if not path.is_file():
        return None
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("checkpoint_kind") == PROJECTION_KIND:
        checksums = json.loads(contained_path(target, "dinov2-small/checksums.json").read_text(encoding="utf-8"))
        if registry.get("model") != MODEL_NAME or registry.get("preprocessing") != PREPROCESS or registry.get("embedding_sha256") != {key: checksums[key] for key in ("config.json", "model.safetensors")}:
            raise ValueError("embedding_index_checkpoint_mismatch")
    return projection_for(registry, target)


def embedding(image):
    vector = backbone_embedding(image)
    weight = active_projection()
    if weight is not None:
        if vector.shape[1] != weight.shape[1]:
            raise ValueError("projection_backbone_dimension_mismatch")
        vector = np.ascontiguousarray(vector @ weight.T, dtype="float32")
        faiss.normalize_L2(vector)
    return vector

SUBSET_NUMBER_PREFIXES = {"AR", "BW", "DP", "GG", "H", "HGSS", "RC", "RT",
                          "S", "SH", "SL", "SM", "SV", "SWSH", "TG", "XY"}


@lru_cache(maxsize=1)
def ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1,
                      det_limit_type="min", det_limit_side_len=224, use_cls=False)
    if hasattr(engine, "text_rec"):
        engine.text_rec.rec_batch_num = 10
    return engine

def parse_numbers(texts):
    pairs=[]
    for text in texts:
        upper = text.upper()
        for prefix, num, total in re.findall(r"(?<![A-Z0-9])([A-Z0-9]*?)([A-Z]{0,4}\d{1,3})\s*/\s*(\d{1,3})\b", upper):
            match = re.match(r"^([A-Z]+)(\d{1,3})$", num)
            if match:
                letters, digits = match.group(1), match.group(2)
                clean = letters + digits if not prefix and letters in SUBSET_NUMBER_PREFIXES else digits
            else:
                clean = num
            pairs.append((clean, total))
    unique=list(dict.fromkeys(pairs))
    return {"number":unique[0][0] if len(unique)==1 else None,"total":unique[0][1] if len(unique)==1 else None,"status":"read" if len(unique)==1 else "missing_or_ambiguous"}


def parse_name(texts):
    candidates = []
    for text in texts:
        text = re.sub(r'\b(?:HP\s*\d+|\d+\s*HP)\b', '', text, flags=re.I).strip()
        if (re.search(r'\bevolves\s*(?:from|into)\b|on\s*the\s*stage|\b[Il1]\s*card\b', text, re.I)
                or re.fullmatch(r'(?:BASIC|STAGE\s*\d?|TRAINER|T[RB]AI[A-Z]*|[A-Z]?\s*ENERG?[A-Z]*|POK[EÉ]MON|SUPPORTER|ITEM|STADIUM|(?:POK[EÉ]MON\s*)?TOOL|\d+)', text, re.I)):
            continue
        if re.search(r'[A-Za-z]{3}', text) and len(text) <= 60:
            candidates.append(text)
    return candidates[0] if candidates else None


def _read_number(image, encode_crop=True):
    # shortcut: fixed ROIs until independent validation supports learned text regions.
    started = time.perf_counter()
    crop = image.crop((0, 560, 504, 704))
    engine = ocr_engine()
    results, header = [], []
    if hasattr(engine, "text_rec") and image.size == (504, 704):
        arr = np.asarray(image)
        strips = [
            (arr[668:690, 428:492], [[428, 108], [492, 108], [492, 130], [428, 130]], "bottom"),
            (arr[660:688, 415:485], [[415, 100], [485, 100], [485, 128], [415, 128]], "bottom"),
            (arr[658:686, 45:165], [[45, 98], [165, 98], [165, 126], [45, 126]], "bottom"),
            (arr[658:686, 78:165], [[78, 98], [165, 98], [165, 126], [78, 126]], "bottom"),
            (arr[674:701, 34:155], [[34, 114], [155, 114], [155, 141], [34, 141]], "bottom"),
            (arr[674:701, 66:155], [[66, 114], [155, 114], [155, 141], [66, 141]], "bottom"),
            (arr[40:78, 118:285], [[118, 40], [285, 40], [285, 76], [118, 76]], "header"),
            (arr[22:64, 88:295], [[88, 22], [295, 22], [295, 58], [88, 58]], "header"),
            (arr[48:92, 22:275], [[22, 48], [275, 48], [275, 84], [22, 84]], "header"),
            (arr[88:128, 48:275], [[48, 88], [275, 88], [275, 124], [48, 124]], "header"),
        ]
        rec_res, _ = engine.text_rec([s[0] for s in strips], False)
        bottom_candidates = []
        saw_bottom_slash = False
        for (_, box, kind), (text, score) in zip(strips, rec_res or []):
            score = float(score)
            if kind == "bottom" and text and "/" in text:
                saw_bottom_slash = True
            if not text or score < 0.55:
                continue
            if kind == "bottom":
                pn = parse_numbers([text])
                if pn["number"] is not None and score >= 0.70:
                    bottom_candidates.append((score, pn["number"], pn["total"], [box, text, score]))
            elif score >= 0.85:
                header.append([box, text, score])
        if bottom_candidates:
            bottom_candidates.sort(key=lambda x: x[0], reverse=True)
            best_num, best_total = bottom_candidates[0][1], bottom_candidates[0][2]
            results = [item[3] for item in bottom_candidates if item[1] == best_num and item[2] == best_total]
        if not results and saw_bottom_slash:
            det_b, _ = engine(arr[645:702, :])
            results = det_b or []
    else:
        results, _ = engine(np.array(crop))
        results = results or []
        header, _ = engine(np.array(image.crop((0, 0, 504, 140))))
        header = header or []
    texts = [item[1] for item in results]
    header_texts = [item[1] for item in header]
    names = []
    for item in header:
        parsed = parse_name([item[1]])
        if parsed:
            box = np.asarray(item[0])
            height = float(np.ptp(box[:, 1])) if box.shape == (4, 2) else 0.0
            names.append((height, parsed, float(item[2])))
    name_entry = max(names, key=lambda item: (item[0], item[2])) if names else None
    name = name_entry[1] if name_entry else None
    name_confidence = name_entry[2] if name_entry else None
    codes = []
    languages = []
    for item in results:
        text, confidence = item[1].upper(), float(item[2])
        # Require a separate printed-code token next to a collector-number pair.
        match = re.search(r"\b([A-Z]{2,5}|[A-Z]{1,3}\d{1,2})\s+(?:(EN|FR|DE|ES|IT|PT|JA|KO|ZH)\s+)?[A-Z]{0,3}\d{1,3}\s*/\s*\d{1,3}\b", text)
        if match and match[1] not in {"EN", "FR", "DE", "ES", "IT", "PT", "JA", "KO", "ZH", "ILLUS", "HP"}:
            codes.append((match[1], confidence))
            if match[2]:
                languages.append((match[2].lower(), confidence))
    code = codes[0] if codes and len({item[0] for item in codes}) == 1 else (None, None)
    language = languages[0] if languages and len({item[0] for item in languages}) == 1 else (None, None)
    ocr_ms = (time.perf_counter() - started) * 1000
    encoding_started = time.perf_counter()
    crop_encoded = encoded(crop) if encode_crop else None
    return {"text": texts, **parse_numbers(texts), "name": name, "name_confidence": name_confidence,
            "set_code": code[0], "set_code_confidence": code[1], "language": language[0],
            "language_confidence": language[1], "name_text": header_texts,
            "roi_mode": "fixed_bottom_header_v3", "crop": crop_encoded,
            "timing_ms": {"ocr": ocr_ms, "encoding": (time.perf_counter() - encoding_started) * 1000}}


def read_number(image):
    return _read_number(image, encode_crop=True)


_default_read_number = read_number


def build_index():
    # Existing explicit production rebuild remains pretrained; trained exports are staging-only.
    if active_projection() is not None:
        raise ValueError("projected_index_requires_versioned_staging_export")
    manifests = sorted((DATA / "catalog").glob("*-manifest.json"))
    rows = [row for path in manifests for row in json.loads(path.read_text(encoding="utf-8"))]
    unique = {r["card_id"]: r for r in rows}
    rows = [catalog_metadata(row, DATA) for row in unique.values()]
    if not rows:
        raise ValueError("Import catalog first")
    vectors, hashes = [], []
    for row in rows:
        path = contained_path(DATA / "gallery", row["provider_id"] + ".webp")
        with Image.open(path) as source:
            image = source.convert("RGB")
        vectors.append(embedding(image)[0])
        hashes.append(phash(image))
    index = faiss.IndexFlatIP(len(vectors[0]))
    index.add(np.stack(vectors))
    target = contained_path(DATA, "models")
    target.mkdir(exist_ok=True)
    temp = target / "gallery.faiss.tmp"
    faiss.write_index(index, str(temp))
    temp.replace(target / "gallery.faiss")
    np.save(target / "gallery-phash.npy", np.stack(hashes))
    model_hashes=json.loads((target/"dinov2-small"/"checksums.json").read_text(encoding="utf-8"))
    model_hashes={name:model_hashes[name] for name in ("config.json","model.safetensors")}
    registry = {"embedding_sha256":model_hashes,"phash_sha256":hashlib.sha256((target/"gallery-phash.npy").read_bytes()).hexdigest(),"model": MODEL_NAME, "checkpoint_kind": "pretrained_not_finetuned", "preprocessing": PREPROCESS, "gallery_version": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), "index_sha256": hashlib.sha256((target / "gallery.faiss").read_bytes()).hexdigest(), "count": len(rows), "rows": rows, "pose": "not_available", "roi": "fixed_bottom_header_v3", "calibration": "not_calibrated"}
    registry["identity_fingerprint"] = identity_fingerprint(registry)
    (target / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry

@lru_cache(maxsize=1)
def gallery():
    target = contained_path(DATA, "models")
    registry = json.loads(contained_path(target, "registry.json").read_text(encoding="utf-8"))
    if registry["preprocessing"] != PREPROCESS or registry["model"] != MODEL_NAME:
        raise ValueError("incompatible_index")
    if sha256(contained_path(target, "gallery.faiss")) != registry["index_sha256"]:
        raise ValueError("index_checksum_mismatch")
    if sha256(contained_path(target, "gallery-phash.npy")) != registry.get("phash_sha256"):
        raise ValueError("phash_checksum_mismatch")
    checksums=json.loads(contained_path(target, "dinov2-small/checksums.json").read_text(encoding="utf-8"))
    if registry.get("embedding_sha256") != {name: checksums.get(name) for name in ("config.json", "model.safetensors")}:
        raise ValueError("embedding_index_checkpoint_mismatch")
    for name in ("config.json", "model.safetensors"):
        if sha256(contained_path(target, "dinov2-small/" + name)) != checksums[name]:
            raise ValueError("embedding_checkpoint_checksum_mismatch")
    weight = projection_for(registry, target)
    if registry.get("identity_fingerprint") and registry["identity_fingerprint"] != identity_fingerprint(registry):
        raise ValueError("identity_fingerprint_mismatch")
    index = faiss.read_index(str(contained_path(target, "gallery.faiss")))
    hashes = np.load(contained_path(target, "gallery-phash.npy"), allow_pickle=False)
    rows = registry["rows"]
    if registry.get("count") != len(rows) or index.ntotal != len(rows) or hashes.shape != (len(rows), 63) or len({row["card_id"] for row in rows}) != len(rows):
        raise ValueError("gallery_row_alignment_mismatch")
    config = json.loads(contained_path(target, "dinov2-small/config.json").read_text(encoding="utf-8"))
    dimension = config["hidden_size"]
    if index.metric_type != faiss.METRIC_INNER_PRODUCT or index.d != (weight.shape[0] if weight is not None else dimension):
        raise ValueError("embedding_index_dimension_or_metric_mismatch")
    if weight is not None and weight.shape[1] != dimension:
        raise ValueError("projection_backbone_dimension_mismatch")
    if hashes.dtype != np.uint8 or not np.isin(hashes, [0, 1]).all():
        raise ValueError("invalid_phash_values")
    # Backward-compatible enrichment of existing registries without rewriting active artifacts.
    registry = {**registry, "rows": [catalog_metadata(row, DATA) if row.get("provider_id") else dict(row) for row in rows]}
    registry["identity_fingerprint"] = identity_fingerprint(registry)
    return index, registry, hashes

def rerank(candidates, number, total=None):
    return rank_candidates(candidates, {"number": number, "total": total})


def identify(image, mode="cropped", corners=None, method="full", *, cardcorpus=_GALLERY_UNSET, encode_images=True):
    from .detection import quality
    start = time.perf_counter()
    if method not in ("phash", "embedding", "embedding_ocr", "full"):
        raise ValueError("invalid_identification_method")
    if mode not in ("cropped", "manual_corners"):
        raise ValueError("pose_weight_missing_use_cropped_or_manual_corners")
    if mode == "manual_corners" and corners is None:
        raise ValueError("manual_corners_required")
    flat, points = warp(image, corners)
    warped_at = time.perf_counter()
    image_quality = quality(flat, DEFAULT_POLICY)
    quality_at = time.perf_counter()
    timing = {key: 0.0 for key in ("gallery", "embedding", "retrieval", "ocr", "fusion", "encoding")}
    timing.update(warp=(warped_at-start)*1000, quality=(quality_at-warped_at)*1000)
    ocr = {"number": None, "total": None, "status": "not_run", "roi_mode": "not_run", "text": []}
    registry = {"model": "not_run", "checkpoint_kind": "not_run", "gallery_version": "not_run"}
    candidates, ranked, versions = [], [], None
    if image_quality["usable"]:
        stage = time.perf_counter()
        selected = active_cardcorpus_gallery() if cardcorpus is _GALLERY_UNSET and method != "phash" else cardcorpus
        if selected is not _GALLERY_UNSET and selected is not None and method != "phash":
            timing["gallery"] = (time.perf_counter()-stage)*1000
            search = selected.search(flat, top_k=50)
            candidates = search["candidates"]
            timing.update(search["timing_ms"])
            versions = {**search["model_versions"], "fusion": FUSION_VERSION,
                        "pose": "external_localization", "roi": "not_run"}
        else:
            index, registry, hashes = gallery()
            timing["gallery"] = (time.perf_counter()-stage)*1000
            stage = time.perf_counter()
            if method == "phash":
                distances = np.mean(hashes != phash(flat), axis=1)
                order = np.argsort(distances)[:20]
                pairs = [(int(i), float(1-distances[i])) for i in order]
            else:
                vector = embedding(flat)
                timing["embedding"] = (time.perf_counter()-stage)*1000
                stage = time.perf_counter()
                scores, ids = index.search(vector, min(20, index.ntotal)) if index.ntotal else ([], [])
                pairs = [(int(i), float(score)) for i, score in zip(ids[0], scores[0]) if i >= 0] if index.ntotal else []
            stage = time.perf_counter()
            for i, score in pairs:
                row = registry["rows"][i]
                candidates.append({**{key: row.get(key) for key in ("card_id", "name", "set", "set_id", "number", "official_total", "language", "variant", "set_codes")}, "similarity": score})
            timing["retrieval"] = (time.perf_counter()-stage)*1000
        stage = time.perf_counter()
        if method in ("full", "embedding_ocr"):
            ocr = _read_number(flat, encode_crop=False) if not encode_images and read_number is _default_read_number else read_number(flat)
            crop_encoding = ocr.get("timing_ms", {}).get("encoding", 0)
            timing["encoding"] += crop_encoding
            timing["ocr"] = max(0.0, (time.perf_counter()-stage)*1000-crop_encoding)
        stage = time.perf_counter()
        ranked = rank_candidates(candidates, ocr)
        result = decision(ranked, ocr, DEFAULT_POLICY)
        timing["fusion"] = (time.perf_counter()-stage)*1000
    else:
        result = {"status": "unknown", "reason_codes": ["unusable_image", *image_quality["reasons"]],
                  "margin": None, "margin_score": "fusion_score", "score_type": "uncalibrated_similarity"}
    stage = time.perf_counter()
    warped = encoded(flat) if encode_images else None
    timing["encoding"] += (time.perf_counter()-stage)*1000
    timing["total"] = (time.perf_counter()-start)*1000
    if versions is None:
        versions = {"embedding": registry["model"], "kind": registry["checkpoint_kind"],
                    "gallery": registry["gallery_version"], "fusion": FUSION_VERSION,
                    "projection": registry.get("projection", {}).get("sha256"),
                    "identity_fingerprint": identity_fingerprint(registry),
                    "preprocessing": PREPROCESS, "pose": "not_available"}
    versions["roi"] = ocr["roi_mode"]
    return {**result, "candidates": ranked[:3], "raw_candidates": candidates, "before_ocr": candidates[:3],
            "quality": image_quality, "ocr": ocr, "warped": warped, "corners": points, "mode": mode,
            "model_versions": versions,
            "timing_ms": {key: round(value, 3) for key, value in timing.items()}}
