import base64
import io
import warnings
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

def decode(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("image_size_limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(raw))
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("unsupported_image_format")
            if image.width * image.height > MAX_PIXELS or min(image.size) < 64:
                raise ValueError("image_dimensions_limit")
            image.load()
            clean = ImageOps.exif_transpose(image).convert("RGB")
            clean.info.clear()
            return clean
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("invalid_image") from exc

def encoded(image):
    out = io.BytesIO()
    image.save(out, format="WEBP", quality=85)
    return base64.b64encode(out.getvalue()).decode()

def warp(image, corners=None):
    rgb = np.array(image)
    h, w = rgb.shape[:2]
    if corners is None:
        points = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    else:
        points = np.array(corners, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("invalid_corners")
        if (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any():
            raise ValueError("corners_outside_image")
        if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < w*h*0.02:
            raise ValueError("corners_not_convex_or_too_small")
        if cv2.contourArea(points, oriented=True) <= 0:
            raise ValueError("corner_order_must_be_TL_TR_BR_BL")
    target = np.array([[0, 0], [503, 0], [503, 703], [0, 703]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    out = cv2.warpPerspective(rgb, matrix, (504, 704))
    return Image.fromarray(out), points.tolist()

def phash(image):
    gray = cv2.cvtColor(np.array(image.resize((32, 32))), cv2.COLOR_RGB2GRAY).astype(np.float32)
    block = cv2.dct(gray)[:8, :8].flatten()[1:]
    return (block > np.median(block)).astype(np.uint8)

