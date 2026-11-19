"""Bounded public reference-art caching for validated CardCorpus picture IDs."""
import ipaddress
import logging
import re
import socket
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import monotonic

import httpx
from PIL import Image

IMAGE_HOST = "img.cardcorpus.com"
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
PICTURE_ID = re.compile(r"[0-9a-f]{32}")
IMAGE_URL = re.compile(r"https://img\.cardcorpus\.com/s1/480/([0-9a-f]{32})\.avif")
_CACHE_LOCKS = tuple(Lock() for _ in range(8))
_FETCH_TIMEOUT = 10
log = logging.getLogger(__name__)


def picture_id(details: dict) -> str | None:
    if not isinstance(details, dict):
        return None
    linked = details.get("cardcorpus")
    external_id = linked.get("id") if isinstance(linked, dict) else None
    if details.get("source") != "CardCorpus" and not (
        isinstance(external_id, str) and len(external_id.split("/")) == 3 and all(external_id.split("/"))
    ):
        return None

    original = details.get("original")
    picture = original.get("picture") if isinstance(original, dict) else None
    front = picture.get("front") if isinstance(picture, dict) else None
    ids = [front["image"]] if isinstance(front, dict) and front.get("image") is not None else []
    for candidate in (details.get("picture_id"), linked.get("picture_id") if isinstance(linked, dict) else None):
        if candidate is not None:
            ids.append(candidate)
    if not ids or any(not isinstance(candidate, str) or not PICTURE_ID.fullmatch(candidate) for candidate in ids):
        return None
    if any(candidate != ids[0] for candidate in ids[1:]):
        return None
    image_url = details.get("image_url")
    if image_url is not None:
        match = IMAGE_URL.fullmatch(image_url) if isinstance(image_url, str) else None
        if match is None or match.group(1) != ids[0]:
            return None
    return ids[0]


def _public_addresses() -> list[str]:
    try:
        records = socket.getaddrinfo(IMAGE_HOST, 443, type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(ipaddress.ip_address(row[4][0]) for row in records))
    except (OSError, ValueError):
        return []
    if any(not address.is_global or address.is_multicast or address.is_reserved
           or getattr(address, "scope_id", None) for address in addresses):
        return []
    return [str(address) for address in addresses[:8]]


def _download(ident: str) -> BytesIO | None:
    deadline = monotonic() + _FETCH_TIMEOUT
    addresses = _public_addresses()
    url = httpx.URL(f"https://{IMAGE_HOST}/s1/480/{ident}.avif")
    with httpx.Client(timeout=httpx.Timeout(3), follow_redirects=False, trust_env=False) as client:
        for address in addresses:
            if monotonic() >= deadline:
                return None
            try:
                # Pin the validated address while retaining CardCorpus TLS identity and HTTP host.
                with client.stream("GET", url.copy_with(host=address), extensions={"sni_hostname": IMAGE_HOST},
                                   headers={"Host": IMAGE_HOST, "Accept": "image/avif", "Accept-Encoding": "identity"}) as response:
                    if response.status_code != 200 or response.headers.get("content-encoding", "identity") != "identity":
                        return None
                    length = response.headers.get("content-length")
                    if length is not None and (not length.isascii() or not length.isdecimal() or int(length) > MAX_BYTES):
                        return None
                    raw = BytesIO()
                    for chunk in response.iter_raw():
                        if monotonic() >= deadline or raw.tell() + len(chunk) > MAX_BYTES:
                            return None
                        raw.write(chunk)
                    if length is not None and raw.tell() != int(length):
                        return None
                    raw.seek(0)
                    return raw
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError):
                continue  # A public IPv6 address may not be reachable on the local machine.
            except (httpx.HTTPError, ValueError, OSError):
                return None
    return None


def _cache_webp(raw: BytesIO, path: Path) -> bool:
    temp = None
    try:
        with Image.open(raw, formats=("AVIF", "JPEG", "PNG", "WEBP")) as source:
            if source.width * source.height > MAX_PIXELS:
                return False
            source.load()
            with source.convert("RGB") as image:
                path.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                    temp = Path(stream.name)
                    image.save(stream, "WEBP", quality=86)
                    if stream.tell() > MAX_BYTES:
                        return False
        temp.replace(path)
        return True
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, OSError, ValueError, SyntaxError, MemoryError) as exc:
        log.debug("Catalog image could not be cached: %s", type(exc).__name__)
        return path.is_file()
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def catalog_image(details: dict, root: Path) -> Path | None:
    ident = picture_id(details)
    if ident is None:
        return None
    root = root.resolve()
    cache_root = root / "card-image-cache"
    if not cache_root.resolve().is_relative_to(root):
        return None
    path = cache_root / (ident + ".webp")
    if path.is_file() and path.resolve().is_relative_to(root):
        return path
    # shortcut: locks coalesce misses per process; multiple workers may download the same picture.
    lock = _CACHE_LOCKS[hash(ident) % len(_CACHE_LOCKS)]
    if not lock.acquire(timeout=_FETCH_TIMEOUT):
        return path if path.is_file() and path.resolve().is_relative_to(root) else None
    try:
        if path.is_file() and path.resolve().is_relative_to(root):
            return path
        raw = _download(ident)
        if raw is not None:
            cache_root.mkdir(parents=True, exist_ok=True)
            if cache_root.resolve().is_relative_to(root) and _cache_webp(raw, path):
                return path
    finally:
        lock.release()
    return None
