import io
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select
from cardlens_core.main import app
from cardlens_core.db import Session, Listing, Quote, Scan, User
from cardlens_core.catalog import save_card, now
from test_flow import FIXTURE

def account(email):
    c=TestClient(app)
    c.post("/api/v1/session").raise_for_status()
    r=c.post("/api/v1/auth/register",json={"email":email,"password":"test-password-123456","display_name":"Collector"})
    assert r.status_code==200,r.text
    return c,c.get("/api/v1/me").json()["owner"]

def listing(seller):
    save_card(FIXTURE,now())
    with Session.begin() as db:
        db.add(Scan(id="market-scan",owner=seller,created_at=now(),result={},card_id="fixture:007:en",variant="normal-unlimited",condition="NM",revision=1))
        db.flush()
        db.add(Quote(id="market-quote",scan_id="market-scan",owner=seller,revision=1,created_at=now(),payload={"amount_vnd":100000,"source_at":now()}))
        db.flush()
        db.add(Listing(id="sale",owner=seller,scan_id="market-scan",quote_id="market-quote",card_id="fixture:007:en",variant="normal-unlimited",condition="NM",asking_vnd=100000,status="draft",created_at=now(),details={}))

def test_accounts_photo_publish_race_and_idempotent_escrow():
    seller,seller_id=account("seller@example.com")
    buyer,buyer_id=account("buyer@example.com")
    rival,rival_id=account("rival@example.com")
    listing(seller_id)
    assert buyer.post("/api/v1/listings/sale/publish",json={"asking_vnd":100000}).status_code==404
    assert seller.post("/api/v1/listings/sale/publish",json={"asking_vnd":100000}).status_code==422
    image=io.BytesIO();Image.new("RGB",(120,180),"blue").save(image,"JPEG")
    rear=io.BytesIO();Image.new("RGB",(120,180),"red").save(rear,"JPEG")
    upload={"front":("front.jpg",image.getvalue(),"image/jpeg"),"back":("back.jpg",rear.getvalue(),"image/jpeg")}
    assert buyer.post("/api/v1/listings/sale/photos",files=upload).status_code==404
    assert seller.post("/api/v1/listings/sale/photos",files=upload).status_code==200
    assert seller.post("/api/v1/listings/sale/publish",json={"asking_vnd":100000}).status_code==200
    assert len(buyer.get("/api/v1/market").json())==1
    def reserve(args):
        client,key=args
        return client.post("/api/v1/orders",json={"listing_ids":["sale"],"request_key":key})
    with ThreadPoolExecutor(2) as pool:
        result=list(pool.map(reserve,[(buyer,"buyer-request"),(rival,"rival-request")]))
    assert sorted(r.status_code for r in result)==[200,409]
    win=0 if result[0].status_code==200 else 1
    client=[buyer,rival][win]
    order=result[win].json()["id"]
    assert client.post("/api/v1/orders/"+order+"/pay").status_code==409
    first=client.post("/api/v1/wallet/demo-credit").json()["balance_vnd"]
    assert client.post("/api/v1/wallet/demo-credit").json()["balance_vnd"]==first
    assert seller.post("/api/v1/orders/"+order+"/pay").status_code==404
    assert client.post("/api/v1/orders/"+order+"/pay").status_code==200
    assert client.post("/api/v1/orders/"+order+"/pay").status_code==200
    assert client.get("/api/v1/wallet").json()["balance_vnd"]==first-100000
    assert seller.post("/api/v1/orders/"+order+"/confirm-received").status_code==404
    assert seller.post("/api/v1/orders/"+order+"/ship",json={"simulated_tracking":"DEMO-TRACK"}).status_code==200
    assert client.post("/api/v1/orders/"+order+"/confirm-received").status_code==200
    assert client.post("/api/v1/orders/"+order+"/confirm-received").status_code==200
    assert seller.get("/api/v1/wallet").json()["balance_vnd"]==100000
    assert buyer.get("/api/v1/market").json()==[]
    assert client.get("/api/v1/orders").json()[0]["is_buyer"] is True
    assert seller.get("/api/v1/orders").json()[0]["is_buyer"] is False
    assert "seller" not in client.get("/api/v1/orders").json()[0]["details"]["items"][0]
    assert seller.get("/api/v1/listings/drafts").json()==[]

def test_login_logout_and_private_messages():
    seller,seller_id=account("seller@example.com")
    buyer,buyer_id=account("buyer@example.com")
    outsider,outsider_id=account("other@example.com")
    listing(seller_id)
    assert buyer.post("/api/v1/messages/sale",json={"body":"Hello"}).status_code==404
    with Session.begin() as db:
        db.get(Listing,"sale").status="active"
    assert buyer.post("/api/v1/messages/sale",json={"body":"Hello"}).status_code==200
    assert seller.get("/api/v1/messages/sale").json()[0]["body"]=="Hello"
    assert outsider.get("/api/v1/messages/sale").json()==[]
    assert outsider.post("/api/v1/messages/sale",json={"body":"new public inquiry"}).status_code==200
    assert [m["body"] for m in buyer.get("/api/v1/messages/sale").json()]==["Hello"]
    messages=seller.get("/api/v1/messages/sale").json()
    first=next(m for m in messages if m["body"]=="Hello")
    assert seller.post("/api/v1/messages/sale",json={"body":"ambiguous reply"}).status_code==422
    assert seller.post("/api/v1/messages/sale",json={"body":"private reply","reply_to":first["id"]}).status_code==200
    assert [m["body"] for m in buyer.get("/api/v1/messages/sale").json()]==["Hello","private reply"]
    assert [m["body"] for m in outsider.get("/api/v1/messages/sale").json()]==["new public inquiry"]
    fresh=TestClient(app)
    assert fresh.post("/api/v1/auth/login",json={"email":"seller@example.com","password":"wrong-password-1234"}).status_code==401
    assert fresh.post("/api/v1/auth/login",json={"email":"seller@example.com","password":"test-password-123456"}).status_code==200
    assert fresh.get("/api/v1/me").json()["owner"]==seller_id
    assert fresh.post("/api/v1/auth/logout").status_code==200
    assert fresh.get("/api/v1/me").status_code==401



def test_price_alert_fires_once_and_stays_private():
    from cardlens_core.db import Fx,PriceAlert,Notification
    from cardlens_core.catalog import evaluate_alerts
    seller,seller_id=account("alerts@example.com")
    outsider,_=account("outsider@example.com")
    save_card(FIXTURE,now())
    with Session.begin() as db:
        from datetime import datetime, timezone
        from cardlens_core.db import Price
        for price in db.scalars(select(Price)):
            price.source_at = now()
        db.add(Fx(id="fx",source_at=str(int(datetime.now(timezone.utc).timestamp())),fetched_at=now(),source="fixture",rates={"USD":"1","VND":"25000"}))
    response=seller.post("/api/v1/alerts",json={"card_id":"fixture:007:en","variant":"normal-unlimited","direction":"below","threshold_vnd":300000})
    assert response.status_code==200
    assert evaluate_alerts()==1
    assert evaluate_alerts()==0
    assert len(seller.get("/api/v1/notifications").json())==1
    assert outsider.get("/api/v1/notifications").json()==[]
    with Session.begin() as db:
        alert=db.scalar(select(PriceAlert))
        alert.active=True
        alert.triggered_at=None
    assert evaluate_alerts()==0
    with Session() as db:
        assert len(db.scalars(select(Notification)).all())==1



def test_duplicate_photos_require_independent_moderator_and_audit():
    from cardlens_core.db import RoleGrant,Audit
    seller,seller_id=account("seller@example.com")
    reviewer,reviewer_id=account("reviewer@example.com")
    outsider,_=account("other@example.com")
    listing(seller_id)
    image=io.BytesIO();Image.new("RGB",(120,180),"green").save(image,"PNG")
    upload={side:(side+".png",image.getvalue(),"image/png") for side in ("front","back")}
    assert seller.post("/api/v1/listings/sale/photos",files=upload).status_code==200
    r=seller.post("/api/v1/listings/sale/publish",json={"asking_vnd":100000})
    assert r.json()["status"]=="pending_review"
    assert outsider.get("/api/v1/admin/moderation").status_code==403
    with Session.begin() as db:
        db.add(RoleGrant(owner=reviewer_id,role="moderator"))
        db.add(RoleGrant(owner=seller_id,role="moderator"))
    assert len(reviewer.get("/api/v1/admin/moderation").json())==1
    assert reviewer.get("/api/v1/listings/sale/photos/front").status_code==200
    assert outsider.get("/api/v1/listings/sale/photos/front").status_code==404
    assert seller.post("/api/v1/admin/moderation/sale",json={"action":"approve","reason":"self approve"}).status_code==403
    assert reviewer.post("/api/v1/admin/moderation/sale",json={"action":"reject","reason":"same image front and back"}).json()["status"]=="removed"
    with Session() as db:
        assert len(db.scalars(select(Audit)).all())==1



def test_source_only_variant_can_be_selected_but_remains_unpriced():
    from cardlens_core.db import Card, Session
    seller, _ = account("source-variant@example.com")
    with Session.begin() as db:
        db.add(Card(id="source:one:en", provider_id="source-one", set_id="source", number="1",
                    language="en", name="Source card", image="", details={
                        "source": "CardCorpus", "variants_detailed": [],
                        "cardcorpus": {"id": "international/example/1",
                                       "source_variant_options": [{"id": "etched", "type": "etched"}]},
                    }))

    response = seller.post("/api/v1/listings/manual", json={
        "card_id": "source:one:en", "variant": "etched", "condition": "NM", "asking_vnd": 500,
    })

    assert response.status_code == 200, response.text
    with Session() as db:
        listing = db.query(Listing).filter_by(card_id="source:one:en").one()
        assert listing.variant == "etched"
        assert db.get(Quote, listing.quote_id).payload["amount_vnd"] is None

def test_manual_listing_and_collection_private_snapshot():
    seller,_=account("collector@example.com")
    outsider,_=account("other@example.com")
    save_card(FIXTURE,now())
    body={"card_id":"fixture:007:en","variant":"normal-unlimited"}
    r=seller.post("/api/v1/listings/manual",json={**body,"asking_vnd":150000})
    assert r.status_code==200
    assert len(seller.get("/api/v1/listings/drafts").json())==1
    assert outsider.get("/api/v1/listings/drafts").json()==[]
    r=seller.post("/api/v1/personal",json={**body,"kind":"collection","quantity":2})
    ident=r.json()["id"]
    assert outsider.delete("/api/v1/personal/"+ident).status_code==404
    value=seller.get("/api/v1/collection/value").json()
    assert value["missing_price_quantity"]==2 and len(value["history"])==1
    assert seller.delete("/api/v1/personal/"+ident).status_code==200


def test_chat_websocket_owner_and_origin_enforcement():
    from starlette.websockets import WebSocketDisconnect
    import pytest
    seller,seller_id=account("seller@example.com")
    buyer,buyer_id=account("buyer@example.com")
    outsider,_=account("outsider@example.com")
    listing(seller_id)
    with Session.begin() as db:db.get(Listing,"sale").status="active"
    buyer.post("/api/v1/messages/sale",json={"body":"hello realtime"})
    thread=buyer.get("/api/v1/messages/sale").json()[0]["thread_id"]
    path="/api/v1/ws/chat/sale/"+thread
    with buyer.websocket_connect(path,headers={"origin":"http://127.0.0.1:3000"}) as ws:
        value=ws.receive_json()
        assert value["type"]=="chat.messages" and value["messages"][0]["body"]=="hello realtime"
        buyer.post("/api/v1/auth/logout")
        ws.send_text("ping")
        with pytest.raises(WebSocketDisconnect):ws.receive_json()
    with pytest.raises(WebSocketDisconnect):
        with outsider.websocket_connect(path,headers={"origin":"http://127.0.0.1:3000"}):pass
    with pytest.raises(WebSocketDisconnect):
        with seller.websocket_connect(path,headers={"origin":"https://evil.example"}):pass


def test_expired_unpaid_order_releases_listing_without_touching_money():
    from datetime import datetime,timedelta,timezone
    from cardlens_core.db import Order
    seller,seller_id=account("seller@example.com")
    buyer,_=account("buyer@example.com")
    listing(seller_id)
    with Session.begin() as db:db.get(Listing,"sale").status="active"
    r=buyer.post("/api/v1/orders",json={"listing_ids":["sale"],"request_key":"expiry-test"})
    ident=r.json()["id"]
    with Session.begin() as db:db.get(Order,ident).created_at=(datetime.now(timezone.utc)-timedelta(minutes=16)).isoformat()
    assert len(buyer.get("/api/v1/market").json())==1
    assert buyer.post("/api/v1/orders/"+ident+"/pay").status_code==409
    assert buyer.get("/api/v1/wallet").json()["balance_vnd"]==0



def test_dispute_freezes_escrow_and_independent_refund_is_idempotent():
    from cardlens_core.db import RoleGrant
    seller,seller_id=account("seller@example.com")
    buyer,buyer_id=account("buyer@example.com")
    reviewer,reviewer_id=account("reviewer@example.com")
    listing(seller_id)
    with Session.begin() as db:
        db.get(Listing,"sale").status="active"
        db.add(RoleGrant(owner=reviewer_id,role="moderator"))
        db.add(RoleGrant(owner=seller_id,role="moderator"))
    buyer.post("/api/v1/wallet/demo-credit")
    ident=buyer.post("/api/v1/orders",json={"listing_ids":["sale"],"request_key":"dispute-request"}).json()["id"]
    buyer.post("/api/v1/orders/"+ident+"/pay").raise_for_status()
    assert buyer.post("/api/v1/orders/"+ident+"/confirm-received").status_code==409
    assert buyer.post("/api/v1/orders/"+ident+"/ship",json={"simulated_tracking":"DEMO-123"}).status_code==404
    assert seller.post("/api/v1/orders/"+ident+"/ship",json={"simulated_tracking":"DEMO-123"}).json()["status"]=="shipped"
    assert buyer.post("/api/v1/orders/"+ident+"/dispute",json={"reason":"wrong card received in simulated order"}).json()["status"]=="disputed"
    assert buyer.post("/api/v1/orders/"+ident+"/confirm-received").status_code==409
    decision={"action":"refund","reason":"confirmed simulated wrong card; refund buyer"}
    assert seller.post("/api/v1/admin/disputes/"+ident+"/resolve",json=decision).status_code==403
    assert reviewer.post("/api/v1/admin/disputes/"+ident+"/resolve",json=decision).json()["status"]=="refunded"
    assert reviewer.post("/api/v1/admin/disputes/"+ident+"/resolve",json=decision).status_code==200
    assert buyer.get("/api/v1/wallet").json()["balance_vnd"]==100000000
    assert seller.get("/api/v1/wallet").json()["balance_vnd"]==0
    assert reviewer.post("/api/v1/admin/disputes/"+ident+"/resolve",json={**decision,"action":"release"}).status_code==409




def test_price_alert_does_not_trigger_from_stale_price_or_fx():
    from cardlens_core.db import Fx,Price,PriceAlert,Notification
    from cardlens_core.catalog import evaluate_alerts
    _,user=account("stale-alert@example.com")
    save_card(FIXTURE,now())
    with Session.begin() as db:
        price=db.scalar(select(Price))
        price.source_at="2020-01-01T00:00:00+00:00"
        db.add(Fx(id="old-fx",source_at="1577836800",fetched_at=now(),source="fixture",rates={"USD":"1","VND":"25000"}))
        db.add(PriceAlert(id="old-alert",owner=user,card_id="fixture:007:en",variant="normal-unlimited",direction="below",threshold_vnd=300000,active=True,triggered_at=None))
    assert evaluate_alerts()==0
    with Session.begin() as db:
        db.scalar(select(Price)).source_at=now()
    assert evaluate_alerts()==0
    with Session.begin() as db:
        import time
        db.get(Fx,"old-fx").source_at=str(int(time.time()))
    assert evaluate_alerts()==1
    with Session() as db:
        assert len(db.scalars(select(Notification)).all())==1
