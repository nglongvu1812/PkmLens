import io
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select, func
from cardlens_core.main import app
from cardlens_core.db import Session, Card, Mapping, Scan, Price, Fx, Listing
from cardlens_core.catalog import save_card, now
from cardlens_core.pricing import calculate
from cardlens_ai.imaging import decode, warp
from cardlens_ai.pipeline import rerank

FIXTURE = {"id":"fixture-007","localId":"007","name":"Test card","set":{"id":"fixture"},"variants_detailed":[{"variantId":"normal-unlimited","type":"normal","subtype":"unlimited","thirdParty":{"tcgplayer":4242},"pricing":{"tcgplayer":{"unit":"USD","updated":"2026-11-14T00:00:00Z","normal":{"marketPrice":10,"lowPrice":8,"highPrice":12}}}},{"variantId":"reverse-unlimited","type":"reverse","subtype":"unlimited"}]}

def start():
    client = TestClient(app)
    assert client.post("/api/v1/session").status_code == 200
    save_card(FIXTURE, now())
    from cardlens_core.main import owner
    from starlette.requests import Request
    request=Request({"type":"http","headers":[(b"cookie", ("pkmlens_session="+client.cookies.get("pkmlens_session")).encode())]})
    ident=owner(request)
    with Session.begin() as db:
        db.add(Fx(id="fx", source_at="1794614400", fetched_at=now(), source="test_fixture", rates={"USD":"1","VND":"25000"}))
        db.add(Mapping(id="fixture-map-normal",card_id="fixture:007:en",provider="tcgplayer",provider_id="fixture-market",variant="normal-unlimited",status="matched"))
        db.add(Scan(id="scan", owner=ident, created_at=now(), result={}, condition="NM", revision=0))
    return client

def confirm(client, variant="normal-unlimited"):
    return client.post("/api/v1/scan/scan/confirm", json={"card_id":"fixture:007:en","variant":variant,"condition":"LP"})

def test_confirm_reprice_stale_quote_and_private_draft():
    client=start()
    first=confirm(client).json()
    assert first["price"]["amount_vnd"] == 212500
    missing=confirm(client,"reverse-unlimited").json()
    assert missing["price"]["amount_vnd"] is None
    assert client.post("/api/v1/listings/from-scan/scan",json={"quote_id":first["quote_id"]}).status_code == 409
    other=TestClient(app)
    other.post("/api/v1/session")
    assert confirm(other).status_code == 404
    assert other.post("/api/v1/listings/from-scan/scan",json={"quote_id":missing["quote_id"]}).status_code == 404
    assert other.get("/api/v1/listings/drafts").json() == []
    response=client.post("/api/v1/listings/from-scan/scan",json={"quote_id":missing["quote_id"],"asking_vnd":120000})
    assert response.status_code==200
    value=response.json()
    assert value["variant"] == "reverse-unlimited" and value["status"]=="draft"
    assert client.post("/api/v1/listings/from-scan/scan",json={"quote_id":missing["quote_id"],"asking_vnd":120000}).json()["id"]==value["id"]
    assert confirm(client).status_code == 409
    assert client.post("/api/v1/listings/from-scan/scan",json={"quote_id":missing["quote_id"],"asking_vnd":-1}).status_code == 422
    with Session() as db:
        assert db.scalar(select(func.count()).select_from(Listing)) == 1

def test_import_is_idempotent_and_preserves_leading_zeros():
    save_card(FIXTURE,now())
    save_card(FIXTURE,now())
    with Session() as db:
        assert db.scalar(select(func.count()).select_from(Card))==1
        assert db.scalar(select(func.count()).select_from(Price))==1
        assert db.get(Card,"fixture:007:en").number=="007"

def test_untrusted_upload_and_origin():
    client=start()
    assert client.post("/api/v1/scan/identify",files={"image":("fake.jpg",b"not a photo","image/jpeg")}).status_code==422
    assert client.post("/api/v1/session",headers={"Origin":"https://evil.invalid"}).status_code==403
    anonymous=TestClient(app)
    assert anonymous.get("/api/v1/listings/drafts").status_code==401
    assert anonymous.post("/api/v1/scan/identify",files={"image":("fake.jpg",b"bad","image/jpeg")}).status_code==401

def test_money_and_image_validation():
    assert calculate("0.00002","25000","NM")==1
    assert calculate("10.10","25000","LP")==214625
    image=Image.new("RGB",(120,180),"red")
    out=io.BytesIO();image.save(out,"PNG")
    assert decode(out.getvalue()).size == (120,180)
    result,points=warp(image,[[0,0],[119,0],[119,179],[0,179]])
    assert result.size==(504,704)
    with pytest.raises(ValueError):
        warp(image,[[0,0],[119,179],[119,0],[0,179]])
    with pytest.raises(ValueError):
        decode(b"broken")

def test_ocr_reranking_missing_evidence_is_neutral():
    candidates=[{"card_id":"a:4:en","similarity":.9},{"card_id":"b:6:en","similarity":.88}]
    assert rerank(candidates,None)[0]["card_id"]=="a:4:en"
    assert rerank(candidates,"6")[0]["card_id"]=="b:6:en"


def test_chunked_body_limit_before_parser():
    import asyncio
    from cardlens_core.limits import BodyLimit
    events=[{"type":"http.request","body":b"1234","more_body":True},{"type":"http.request","body":b"5678","more_body":False}]
    sent=[]
    async def receive():
        return events.pop(0)
    async def send(event):
        sent.append(event)
    async def endpoint(scope,receive,send):
        raise AssertionError("Oversize body reached endpoint")
    asyncio.run(BodyLimit(endpoint,maximum=6)({"type":"http","method":"POST"},receive,send))
    assert sent[0]["status"]==413


def test_ambiguous_provider_price_mapping_is_not_used():
    from cardlens_core.db import Mapping
    detail={"id":"same-art-4","localId":"4","name":"Twin","set":{"id":"same"},"variants_detailed":[{"variantId":"holo-one","type":"holo","thirdParty":{"tcgplayer":999},"pricing":{"tcgplayer":{"unit":"USD","updated":"2026-11-14","holofoil":{"marketPrice":25}}}},{"variantId":"holo-two","type":"holo","thirdParty":{"tcgplayer":999},"pricing":{"tcgplayer":{"unit":"USD","updated":"2026-11-14","holofoil":{"marketPrice":25}}}}]}
    save_card(detail,now())
    with Session() as db:
        mappings=db.scalars(select(Mapping).where(Mapping.card_id=="same:4:en")).all()
        assert len([m for m in mappings if m.provider=="tcgplayer"])==2 and all(m.status=="unmatched" for m in mappings if m.provider=="tcgplayer")
        assert db.scalar(select(func.count()).select_from(Price).where(Price.card_id=="same:4:en"))==0
        from cardlens_core.pricing import quote_payload
        quote=quote_payload(db,db.get(Card,"same:4:en"),"holo-one","NM")
    assert quote["amount_vnd"] is None and quote["reason"]=="missing_variant_price"



def test_provider_id_shared_between_cards_is_unmatched_and_never_quoted():
    from cardlens_core.catalog import reconcile_provider_mappings
    first={"id":"same-a","localId":"1","name":"First","set":{"id":"same-a"},"variants_detailed":[{"variantId":"shared-one","type":"holo","thirdParty":{"tcgplayer":555},"pricing":{"tcgplayer":{"unit":"USD","updated":"2026-11-14","holofoil":{"marketPrice":50}}}}]}
    second={"id":"same-b","localId":"1","name":"Second","set":{"id":"same-b"},"variants_detailed":[{"variantId":"shared-two","type":"holo","thirdParty":{"tcgplayer":555},"pricing":{"tcgplayer":{"unit":"USD","updated":"2026-11-14","holofoil":{"marketPrice":50}}}}]}
    save_card(first,now());save_card(second,now())
    assert reconcile_provider_mappings()==1
    from cardlens_core.pricing import quote_payload
    with Session() as db:
        quote=quote_payload(db,db.get(Card,"same-a:1:en"),"shared-one","NM")
    assert quote["amount_vnd"] is None and quote["reason"]=="missing_variant_price"


def test_ocr_pairs_preserve_leading_zero_and_disambiguate_set_total():
    from cardlens_ai.pipeline import parse_numbers
    assert parse_numbers(["007/102"])=={"number":"007","total":"102","status":"read"}
    assert parse_numbers(["4/102","6/130"])["number"] is None
    candidates=[{"card_id":"a:7:en","similarity":.9,"official_total":"102"},{"card_id":"b:7:en","similarity":.88,"official_total":"130"}]
    assert rerank(candidates,"7","130")[0]["card_id"]=="b:7:en"
    assert rerank([{"card_id":"a:007:en","similarity":.9}],"7")[0]["evidence"]["number_match"] is False


def test_price_snapshot_freshness_and_invalid_fx_are_explicit():
    client=start()
    with Session.begin() as db:
        price=db.scalar(select(Price))
        price.source_at="2020-01-01T00:00:00Z"
    quote=confirm(client).json()["price"]
    assert quote["amount_vnd"]==212500 and quote["price_stale"] is True
    with Session.begin() as db:
        fx=db.get(Fx,"fx");fx.rates={"USD":"0","VND":"25000"}
    quote=confirm(client).json()["price"]
    assert quote["amount_vnd"] is None and quote["reason"]=="invalid_price_or_fx_snapshot"



def test_price_refresh_reports_secondary_failure_without_losing_primary_snapshot(monkeypatch):
    import httpx
    from cardlens_core import catalog
    response=httpx.Response(401,request=httpx.Request("GET","https://tcgcsv.test/prices"))
    save_card(FIXTURE,now())
    with Session.begin() as db:
        db.add(Card(id="cc-japanese-example:1:ja", provider_id="cc-example", set_id="cc-japanese-example",
                    number="1", language="ja", name="CardCorpus-only", image="", details={"source": "CardCorpus"}))
    calls = []
    def sync_set(set_id, *, refresh, provider_ids):
        calls.append((set_id, refresh, set(provider_ids)))
        return 3
    monkeypatch.setattr(catalog,"sync_set",sync_set)
    monkeypatch.setattr(catalog,"sync_fx",lambda:"1794700952")
    monkeypatch.setattr(catalog,"evaluate_alerts",lambda:0)
    monkeypatch.setattr("cardlens_core.tcgcsv.sync_tcgcsv",lambda:(_ for _ in ()).throw(httpx.HTTPStatusError("unauthorized",request=response.request,response=response)))
    result=catalog.refresh_prices()
    assert calls == [("fixture", True, {"fixture-007"})]
    assert result=={"cards":3,"fx":"1794700952","tcgcsv":{"status":"unavailable","http_status":401,"error":"upstream_request_failed"},"alerts_triggered":0}


def test_tcgdex_refresh_keeps_attached_cardcorpus_options():
    card_id = save_card(FIXTURE, now())
    evidence = {"id": "international/example/007", "source_variant_options": [{"id": "sourced-holo"}]}
    with Session.begin() as db:
        card = db.get(Card, card_id)
        card.details = {**card.details, "cardcorpus": evidence, "cardcorpus_id": evidence["id"],
                        "number_is_printed": True}

    save_card(FIXTURE, now())

    with Session() as db:
        card = db.get(Card, card_id)
        assert card.details["cardcorpus"] == evidence
        assert card.details["cardcorpus_id"] == evidence["id"]
        assert card.details["number_is_printed"] is True
