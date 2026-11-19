import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import event

from cardlens_core import catalog_images, main, shop
from cardlens_core.db import Card, Session, engine

PICTURE = "0123456789abcdef" * 2
URL = f"https://img.cardcorpus.com/s1/480/{PICTURE}.avif"
HTTP_CLIENT = httpx.Client


def add_card(card_id, *, set_id="set", language="en", provider_id=None, details=None, image=""):
    if details is None:
        details = {"set": {"name": f"{set_id} ({language})"}}
    with Session.begin() as db:
        db.add(Card(id=card_id, provider_id=provider_id or card_id, set_id=set_id, number="1",
                    language=language, name=card_id, image=image, details=details))


def corpus_details(ident=PICTURE, **fields):
    return {"source": "CardCorpus", "original": {"picture": {"front": {"image": ident}}}, **fields}


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(shop, "DATA", tmp_path)
    return TestClient(main.app)


def addresses(*ips):
    return [(0, 0, 0, "", (ip, 443)) for ip in ips]


@pytest.fixture
def fake_upstream(monkeypatch):
    monkeypatch.setattr(catalog_images.socket, "getaddrinfo", lambda *_a, **_k: addresses("93.184.216.34"))

    def install(handler):
        def client(**kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False
            return HTTP_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(catalog_images.httpx, "Client", client)

    return install


def art_bytes():
    output = BytesIO()
    Image.new("RGB", (40, 60), "orange").save(output, "AVIF")
    return output.getvalue()


class ImageStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        return iter(self.chunks)


def test_sets_aggregate_once_with_one_option_per_id_and_separate_printings(api):
    add_card("en2", set_id="shared", details={"set": {"name": "A English"}})
    add_card("en1", set_id="shared", details={"set": {"name": "Z English"}})
    add_card("ja", set_id="shared", language="ja")
    add_card("no-title", set_id="untitled", details={})
    statements = []

    def record(_conn, _cursor, statement, _parameters, _context, _many):
        if "cards" in statement.lower():
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        response = api.get("/api/v1/shop/sets")
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert response.status_code == 200
    assert response.json() == [
        {"id": "shared", "name": "Z English", "count": 3, "language": "en"},
        {"id": "untitled", "name": "untitled", "count": 1, "language": "en"},
    ]
    assert len(statements) == 1
    full_set = api.get("/api/v1/shop/catalog", params={"set_id": "shared", "limit": 2})
    assert full_set.json()["total"] == 3 and len(full_set.json()["items"]) == 2
    response = api.get("/api/v1/shop/catalog", params={"set_id": "shared", "language": "ja"})
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["id"] == "ja"


def test_catalog_coverage_counts_live_printings_distinct_variants_and_valid_source_art(api):
    corpus = {"source": "CardCorpus", "picture_id": PICTURE,
              "original": {"picture": {"front": {"image": PICTURE}}},
              "variants_detailed": [{"variantId": "normal"}, {"variantId": "normal"}, {"variantId": "reverse"}]}
    add_card("coverage-card-1", details=corpus)
    add_card("coverage-card-2", details=corpus_details("../invalid"))
    add_card("coverage-legacy", details={"source": "TCGdex", "variants": {"normal": True, "holo": True}})

    response = api.get("/api/v1/shop/catalog/coverage")

    assert response.status_code == 200
    assert response.json() == {
        "source": "live_database", "total_printings": 3, "total_variants": 4,
        "total_with_source_front_image": 1,
        "by_source": {
            "CardCorpus": {"printings": 2, "variants": 2, "front_images": 1},
            "TCGdex": {"printings": 1, "variants": 2, "front_images": 0},
        },
        "large_index_available": False, "large_index_active": False, "candidate_source": None,
    }


def test_existing_local_art_takes_priority(api, tmp_path, monkeypatch):
    add_card("local-card", details=corpus_details(image_url=URL))
    local = tmp_path / "gallery" / "local-card.webp"
    local.parent.mkdir()
    local.write_bytes(b"existing-local-art")
    monkeypatch.setattr(main, "catalog_image", lambda *_: pytest.fail("must serve the gallery first"))

    response = api.get("/api/v1/cards/local-card/image")

    assert response.status_code == 200 and response.content == b"existing-local-art"
    assert response.headers["content-type"] == "image/webp"
    assert local.read_bytes() == b"existing-local-art"


def test_gallery_path_does_not_escape(api, tmp_path):
    add_card("bad-provider-card", provider_id="../private")
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    (tmp_path / "private.webp").write_bytes(b"outside the gallery")
    (gallery / "private.webp").symlink_to(tmp_path / "private.webp")

    assert api.get("/api/v1/cards/bad-provider-card/image").status_code == 404


@pytest.mark.parametrize("details", [
    {}, {"source": "CardCorpus", "image_url": URL}, corpus_details("../private"),
    corpus_details("A" * 32), corpus_details(PICTURE + "\n"),
    corpus_details(image_url="http://127.0.0.1/private"), corpus_details(image_url=URL + "?redirect=1"),
    corpus_details(image_url=URL.replace("img.cardcorpus.com", "img.cardcorpus.com.evil.invalid")),
    corpus_details(image_url=URL.replace("img.cardcorpus.com", "img.cardcorpus.com@evil.invalid")),
    corpus_details(image_url=URL.replace("/480/", "/960/")),
    corpus_details(image_url=URL.replace(PICTURE, "a" * 32)),
    {"cardcorpus": {"id": "international/example/1"}, "image_url": URL},
    {"cardcorpus": {"id": "international/example/1", "picture_id": "../bad"}},
])
def test_unvalidated_picture_returns_404_without_dns(api, monkeypatch, details):
    add_card("invalid-art", details=details, image=URL)
    monkeypatch.setattr(catalog_images.socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("unexpected DNS"))

    assert api.get("/api/v1/cards/invalid-art/image").status_code == 404
    assert api.get("/api/v1/cards/not-in-db/image").status_code == 404


def test_corpus_art_is_converted_and_cached_for_shared_pictures(api, tmp_path, fake_upstream):
    details = corpus_details(image_url=URL)
    add_card("uncached-art", details=details, image="http://127.0.0.1/never-used")
    add_card("same-picture", details={**details, "source": "TCGdex", "original": {},
                                     "cardcorpus": {"id": "international/example/1", "picture_id": PICTURE}})
    encoded = art_bytes()
    requests = []

    def upstream(request):
        requests.append(request)
        return httpx.Response(200, stream=ImageStream([encoded]))

    fake_upstream(upstream)
    response = api.get("/api/v1/cards/uncached-art/image")
    again = api.get("/api/v1/cards/same-picture/image")

    assert response.status_code == again.status_code == 200
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == URL.replace("img.cardcorpus.com", "93.184.216.34")
    assert request.headers["host"] == "img.cardcorpus.com"
    assert request.extensions["sni_hostname"] == "img.cardcorpus.com"
    path = tmp_path / "card-image-cache" / (PICTURE + ".webp")
    assert path.read_bytes() == response.content == again.content
    with Image.open(path) as cached:
        assert (cached.format, cached.mode, cached.size) == ("WEBP", "RGB", (40, 60))


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("location", [URL, "http://127.0.0.1/private", "https://elsewhere.invalid/image"])
def test_redirect_is_not_followed(api, tmp_path, fake_upstream, location, status):
    add_card("redirect-card", details=corpus_details())
    requests = []

    def upstream(request):
        requests.append(request)
        return httpx.Response(status, headers={"Location": location})

    fake_upstream(upstream)

    assert api.get("/api/v1/cards/redirect-card/image").status_code == 404
    assert len(requests) == 1
    assert not list(tmp_path.rglob("*.webp"))


def test_private_dns_result_blocks_all_addresses(api, monkeypatch, fake_upstream):
    add_card("private-dns", details=corpus_details())
    fake_upstream(lambda _request: pytest.fail("must not connect when any address is private"))
    monkeypatch.setattr(catalog_images.socket, "getaddrinfo",
                        lambda *_a, **_k: addresses("93.184.216.34", "127.0.0.1"))

    assert api.get("/api/v1/cards/private-dns/image").status_code == 404


@pytest.mark.parametrize("failure", ["content_length", "stream_size", "compressed", "corrupt", "pixels", "timeout", "truncated", "wrong_length", "not_found"])
def test_invalid_upstream_does_not_cache_or_change_existing_art(api, tmp_path, monkeypatch, fake_upstream, failure):
    add_card("invalid-upstream", details=corpus_details())
    existing = tmp_path / "card-image-cache" / ("f" * 32 + ".webp")
    existing.parent.mkdir()
    existing.write_bytes(b"already cached")

    def upstream(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("mock read timeout", request=request)
        if failure == "content_length":
            return httpx.Response(200, headers={"Content-Length": str(catalog_images.MAX_BYTES + 1)})
        if failure == "stream_size":
            chunk = b"x" * (1024 * 1024)
            return httpx.Response(200, stream=ImageStream([chunk] * 9))
        if failure == "not_found":
            return httpx.Response(404)
        if failure == "wrong_length":
            return httpx.Response(200, headers={"Content-Length": "0"}, stream=ImageStream([art_bytes()]))
        if failure == "truncated":
            return httpx.Response(200, stream=ImageStream([art_bytes()[:40]]))
        if failure == "compressed":
            return httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=ImageStream([b"x"]))
        if failure == "pixels":
            monkeypatch.setattr(catalog_images, "MAX_PIXELS", 2399)
            return httpx.Response(200, stream=ImageStream([art_bytes()]))
        return httpx.Response(200, stream=ImageStream([b"not an image"]))

    fake_upstream(upstream)

    assert api.get("/api/v1/cards/invalid-upstream/image").status_code == 404
    assert list(existing.parent.iterdir()) == [existing]
    assert existing.read_bytes() == b"already cached"


def test_concurrent_image_cache_misses_share_a_download(tmp_path, fake_upstream):
    started, release = Event(), Event()
    requests = []
    encoded = art_bytes()

    def upstream(request):
        requests.append(request)
        started.set()
        assert release.wait(5)
        return httpx.Response(200, stream=ImageStream([encoded]))

    fake_upstream(upstream)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(catalog_images.catalog_image, corpus_details(), tmp_path)
        try:
            assert started.wait(5)
            second = pool.submit(catalog_images.catalog_image, corpus_details(), tmp_path)
        finally:
            release.set()
        assert first.result(5) == second.result(5) == tmp_path / "card-image-cache" / (PICTURE + ".webp")
    assert len(requests) == 1


def test_cached_picture_is_served_without_network(api, tmp_path, monkeypatch):
    add_card("cached-art", details=corpus_details())
    path = tmp_path / "card-image-cache" / (PICTURE + ".webp")
    path.parent.mkdir()
    path.write_bytes(b"existing-cache-art")
    monkeypatch.setattr(catalog_images.socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("cache hit must not resolve DNS"))

    response = api.get("/api/v1/cards/cached-art/image")

    assert response.status_code == 200 and response.content == b"existing-cache-art"
    assert path.read_bytes() == b"existing-cache-art"


def test_atomic_cache_replace_failure_keeps_existing_image(tmp_path, monkeypatch):
    path = tmp_path / "existing.webp"
    path.write_bytes(b"old-cache")

    def cannot_replace(self, _target):
        raise PermissionError("mock replacement failure")

    monkeypatch.setattr(type(path), "replace", cannot_replace)

    assert catalog_images._cache_webp(BytesIO(art_bytes()), path)
    assert path.read_bytes() == b"old-cache"
    assert list(tmp_path.iterdir()) == [path]


def test_linked_corpus_picture_is_accepted_only_for_valid_source_ids(api, fake_upstream):
    details = {"cardcorpus": {"id": "international/example/1", "picture_id": PICTURE},
               "image_url": URL, "original": {}}
    add_card("linked-corpus-card", details=details)
    encoded = art_bytes()
    requests = []

    def upstream(request):
        requests.append(request)
        return httpx.Response(200, stream=ImageStream([encoded]))

    fake_upstream(upstream)
    response = api.get("/api/v1/cards/linked-corpus-card/image")

    assert response.status_code == 200 and len(requests) == 1
    assert str(requests[0].url) == URL.replace("img.cardcorpus.com", "93.184.216.34")


def test_catalog_and_card_view_expose_only_unverified_corpus_source_variants(api):
    details = {"source": "CardCorpus", "set": {"name": "Source set"},
               "variants_detailed": [{"variantId": "local-holo", "type": "holo", "subtype": "unlimited"}],
               "cardcorpus": {"id": "international/example/1",
                              "source_variant_options": ["local-holo", "unlisted-reverse", "bad\noption", 7,
                                                         {"id": "etched", "label": "Etched", "type": "etched"}]}}
    add_card("source-options", details=details)

    catalog = api.get("/api/v1/shop/catalog", params={"ids": "source-options"}).json()["items"][0]
    detail = api.get("/api/v1/cards/source-options").json()

    expected = [
        {"id": "local-holo", "label": "holo / unlimited", "variant_id": "local-holo", "variant_source": "CardCorpus", "verified": False},
        {"id": "unlisted-reverse", "label": "unlisted-reverse", "type": "source_only", "source": "CardCorpus",
         "variant_id": "unlisted-reverse", "variant_source": "CardCorpus", "verified": False},
        {"id": "etched", "label": "etched", "type": "etched", "source": "CardCorpus",
         "variant_id": "etched", "variant_source": "CardCorpus", "verified": False},
    ]
    assert catalog["variants"] == expected
    assert detail["variants"] == expected


def test_gallery_coverage_requires_the_mapping_checksum(tmp_path, monkeypatch):
    root = tmp_path / "models" / "cardcorpus-scanner"
    root.mkdir(parents=True)
    monkeypatch.setattr(shop, "DATA", tmp_path)
    required = {"pipeline.json", "index/release.json", "index/embeddings.npy", "index/rows.json",
                "index/cards.jsonl.gz", "index/dupe-groups.json", "models/matcher.safetensors",
                "models/heads.safetensors", "mapping.json"}
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")
    digest = __import__("hashlib").sha256((root / "mapping.json").read_bytes()).hexdigest()
    manifest = {name: "0" * 64 for name in required} | {"mapping.json": digest}
    (root / "checksums.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert shop._cardcorpus_gallery_available()
    (root / "models" / "heads.safetensors").unlink()
    assert not shop._cardcorpus_gallery_available()


def test_explicit_source_only_variant_quote_stays_unpriced(api):
    add_card("source-only-price", details={
        "source": "CardCorpus", "cardcorpus": {"id": "international/example/1",
        "source_variant_options": ["etched"]},
    })
    api.post("/api/v1/session").raise_for_status()

    response = api.post("/api/v1/prices/batch", json={"items": [
        {"card_id": "source-only-price", "variant": "etched", "condition": "NM"},
    ]})

    assert response.status_code == 200
    assert response.json()[0]["variant"] == "etched"
    assert response.json()[0]["amount_vnd"] is None
    assert response.json()[0]["reason"] == "missing_variant_price"


def test_catalog_image_cache_rejects_symlink_paths(api, tmp_path, monkeypatch):
    add_card("symlink-art", details=corpus_details())
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    try:
        (tmp_path / "card-image-cache").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    monkeypatch.setattr(catalog_images.socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("unsafe cache path must stop before DNS"))

    assert api.get("/api/v1/cards/symlink-art/image").status_code == 404
    assert not list(outside.iterdir())
