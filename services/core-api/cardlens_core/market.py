import io
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Literal
import boto3
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select, func, update
from .accounts import member, owner
from .db import Session, Listing, Order, Ledger, PersonalCard, Card, Message, DATA, User, Identity
from .catalog import catalog_variants, now, variants

router=APIRouter(prefix="/api/v1")
lock=threading.Lock()

def owned(db,model,ident,user):
    row=db.get(model,ident)
    if not row or row.owner!=user:
        raise HTTPException(404,"not_found")
    return row

def s3():
    return boto3.client("s3",endpoint_url=os.getenv("MINIO_ENDPOINT"),aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),region_name="us-east-1")

def save_photo(raw,key):
    from cardlens_ai.imaging import decode, MAX_BYTES
    image=decode(raw)
    image.thumbnail((1600,1600))
    out=io.BytesIO();image.save(out,"WEBP",quality=90)
    content=out.getvalue()
    if os.getenv("STORAGE_MODE","local")=="minio":
        client=s3();bucket=os.getenv("MINIO_BUCKET","listings")
        try:
            client.head_bucket(Bucket=bucket)
        except client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404","NoSuchBucket"):
                raise
            client.create_bucket(Bucket=bucket)
        client.put_object(Bucket=bucket,Key=key,Body=content,ContentType="image/webp")
    else:
        # shortcut: local development storage; Compose config uses MinIO.
        path=DATA/"listing-images"/key
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
    import hashlib
    from cardlens_ai.imaging import phash
    return {"key":key,"sha256":hashlib.sha256(content).hexdigest(),"phash":"".join(str(int(bit)) for bit in phash(image)),"storage":os.getenv("STORAGE_MODE","local")}

@router.post("/listings/{listing_id}/photos")
def photos(listing_id: str,front: UploadFile=File(),back: UploadFile=File(),user=Depends(member)):
    from cardlens_ai.imaging import decode, MAX_BYTES
    raw_front=front.file.read(MAX_BYTES+1);raw_back=back.file.read(MAX_BYTES+1)
    try:
        decode(raw_front);decode(raw_back)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    with lock, Session.begin() as db:
        row=owned(db,Listing,listing_id,user)
        if row.status!="draft":
            raise HTTPException(409,"listing_not_draft")
        try:
            nonce=secrets.token_hex(8)
            values={side:save_photo(raw,listing_id+"/"+nonce+"-"+side+".webp") for side,raw in (("front",raw_front),("back",raw_back))}
        except Exception as exc:
            raise HTTPException(503,"image_storage_unavailable") from exc
        row.details={**row.details,"photos":values,"training_consent":False}
    return {"status":"photos_saved","training_consent":False}

class Publish(BaseModel):
    asking_vnd: int=Field(ge=1,le=2_000_000_000)

@router.post("/listings/{listing_id}/publish")
def publish(listing_id: str,body: Publish,user=Depends(member)):
    with lock,Session.begin() as db:
        row=owned(db,Listing,listing_id,user)
        if row.status!="draft":
            raise HTTPException(409,"listing_not_draft")
        if set(row.details.get("photos",{}))!={"front","back"}:
            raise HTTPException(422,"front_and_back_photos_required")
        row.asking_vnd=body.asking_vnd
        reasons=[]
        front=row.details["photos"]["front"]
        back=row.details["photos"]["back"]
        if front["sha256"]==back["sha256"]:reasons.append("front_back_identical")
        for other in db.scalars(select(Listing).where(Listing.id!=row.id,Listing.status.in_(["active","reserved","sold","pending_review"]))):
            previous=other.details.get("photos",{}).get("front",{})
            if previous.get("sha256")==front["sha256"]:reasons.append("duplicate_listing_photo")
            elif previous.get("phash") and front.get("phash") and sum(a!=b for a,b in zip(previous["phash"],front["phash"]))<=3:reasons.append("similar_listing_photo_requires_review")
        row.status="pending_review" if reasons else "active"
        row.details={**row.details,"moderation":{"reason_codes":sorted(set(reasons)),"authenticity_verified":False,"declared_card_match":"not_validated"}}
        status=row.status
    return {"id":listing_id,"status":status}

@router.get("/market")
def market(q: str="",set_id: str | None=None,maximum_vnd: int | None=None,user=Depends(owner)):
    with lock, Session.begin() as db:
        expire_reservations(db)
        statement=select(Listing).join(Card).where(Listing.status=="active")
        if q:
            statement=statement.where(Card.name.ilike("%"+q[:100]+"%"))
        if set_id:
            statement=statement.where(Card.set_id==set_id)
        if maximum_vnd is not None:
            statement=statement.where(Listing.asking_vnd<=maximum_vnd)
        rows=db.scalars(statement.order_by(Listing.created_at.desc()).limit(100)).all()
        from .db import Quote
        result=[]
        for row in rows:
            card=db.get(Card,row.card_id)
            quote=db.get(Quote,row.quote_id)
            reference=quote.payload.get("amount_vnd")
            deal="unknown" if not reference else "good" if row.asking_vnd<reference*.9 else "high" if row.asking_vnd>reference*1.1 else "fair"
            seller=db.scalar(select(User).where(User.owner==row.owner))
            result.append({"id":row.id,"card_id":row.card_id,"name":card.name,"variant":row.variant,"condition":row.condition,"asking_vnd":row.asking_vnd,"seller_display_name":seller.display_name if seller else "Collector","is_owner":row.owner==user,"deal":deal,"reference_vnd":reference,"reference_source_at":quote.payload.get("source_at"),"image":"/api/v1/listings/"+row.id+"/photos/front"})
        return result

@router.get("/listings/{listing_id}/photos/{side}")
def photo(listing_id:str,side:Literal["front","back"],user=Depends(owner)):
    with Session() as db:
        row=db.get(Listing,listing_id)
        if not row:raise HTTPException(404)
        if row.status not in ("active","sold","reserved") and row.owner!=user:
            from .db import RoleGrant
            grant=db.get(RoleGrant,user)
            if not grant or grant.role not in ("admin","moderator"):raise HTTPException(404)
        value=row.details.get("photos",{}).get(side)
        if not value:
            raise HTTPException(404)
        try:
            if value["storage"]=="minio":
                content=s3().get_object(Bucket=os.getenv("MINIO_BUCKET","listings"),Key=value["key"])["Body"].read()
            else:
                content=(DATA/"listing-images"/value["key"]).read_bytes()
        except Exception as exc:
            raise HTTPException(503,"image_storage_unavailable") from exc
        return Response(content,media_type="image/webp")

def balance(db,user):
    return db.scalar(select(func.coalesce(func.sum(Ledger.delta_vnd),0)).where(Ledger.owner==user))

@router.get("/wallet")
def wallet(user=Depends(member)):
    with Session() as db:
        return {"balance_vnd":balance(db,user),"simulated":True}

@router.post("/wallet/demo-credit")
def credit(user=Depends(member)):
    with lock,Session.begin() as db:
        db.scalar(select(User).where(User.owner==user).with_for_update())
        if not db.scalar(select(Ledger).where(Ledger.owner==user,Ledger.reference=="demo-credit")):
            db.add(Ledger(id=secrets.token_hex(16),owner=user,delta_vnd=100_000_000,reference="demo-credit",created_at=now()))
    return wallet(user)

class OrderInput(BaseModel):
    listing_ids: list[str]=Field(min_length=1,max_length=20)
    request_key: str=Field(min_length=8,max_length=64)

def order_view(order):
    return {"id":order.id,"status":order.status,"total_vnd":order.total_vnd,"details":{"items":[{"listing_id":item["listing_id"],"card_id":item["card_id"],"price_vnd":item["price_vnd"]} for item in order.details["items"]]},"simulated":True}

@router.post("/orders")
def order(body:OrderInput,user=Depends(member)):
    if len(set(body.listing_ids))!=len(body.listing_ids):
        raise HTTPException(422,"duplicate_listing")
    with lock,Session.begin() as db:
        expire_reservations(db)
        db.scalar(select(User).where(User.owner==user).with_for_update())
        existing=db.scalar(select(Order).where(Order.buyer==user,Order.request_key==body.request_key))
        if existing:
            if sorted(i["listing_id"] for i in existing.details["items"]) != sorted(body.listing_ids):
                raise HTTPException(409,"request_key_payload_mismatch")
            return {**order_view(existing),"is_buyer":True}
        items=[]
        for ident in sorted(body.listing_ids):
            row=db.scalar(select(Listing).where(Listing.id==ident).with_for_update())
            if not row or row.status!="active" or row.owner==user:
                raise HTTPException(409,"listing_unavailable")
            changed=db.execute(update(Listing).where(Listing.id==ident,Listing.status=="active").values(status="reserved"))
            if changed.rowcount!=1:
                raise HTTPException(409,"listing_unavailable")
            items.append({"listing_id":ident,"seller":row.owner,"price_vnd":row.asking_vnd,"card_id":row.card_id})
        row=Order(id=secrets.token_hex(16),buyer=user,request_key=body.request_key,status="created",total_vnd=sum(i["price_vnd"] for i in items),created_at=now(),details={"items":items})
        db.add(row)
        return {**order_view(row),"is_buyer":True}

@router.get("/orders")
def orders(user=Depends(member)):
    with Session() as db:
        rows=db.scalars(select(Order).order_by(Order.created_at.desc())).all()
        return [{**order_view(r),"is_buyer":r.buyer==user} for r in rows if r.buyer==user or any(i["seller"]==user for i in r.details["items"])]

@router.post("/orders/{order_id}/pay")
def pay(order_id:str,user=Depends(member)):
    with lock,Session.begin() as db:
        record=db.scalar(select(Order).where(Order.id==order_id).with_for_update())
        if not record or record.buyer!=user:
            raise HTTPException(404)
        if record.status in ("paid_escrow","shipped","released"):
            return {**order_view(record),"is_buyer":True}
        if record.status!="created":
            raise HTTPException(409,"invalid_order_state")
        if datetime.fromisoformat(record.created_at)<datetime.now(timezone.utc)-timedelta(minutes=15):
            raise HTTPException(409,"reservation_expired")
        db.scalar(select(User).where(User.owner==user).with_for_update())
        if balance(db,user)<record.total_vnd:
            raise HTTPException(409,"insufficient_simulated_balance")
        db.add(Ledger(id=secrets.token_hex(16),owner=user,delta_vnd=-record.total_vnd,reference="pay:"+order_id,created_at=now()))
        record.status="paid_escrow"
        return {**order_view(record),"is_buyer":True}

@router.post("/orders/{order_id}/confirm-received")
def release(order_id:str,user=Depends(member)):
    with lock,Session.begin() as db:
        record=db.scalar(select(Order).where(Order.id==order_id).with_for_update())
        if not record or record.buyer!=user:
            raise HTTPException(404)
        if record.status=="released":
            return {**order_view(record),"is_buyer":True}
        if record.status!="shipped":
            raise HTTPException(409,"order_not_shipped")
        seller_totals={}
        for item in record.details["items"]:
            seller_totals[item["seller"]]=seller_totals.get(item["seller"],0)+item["price_vnd"]
            db.get(Listing,item["listing_id"]).status="sold"
        for seller in sorted(seller_totals):
            db.scalar(select(User).where(User.owner==seller).with_for_update())
            db.add(Ledger(id=secrets.token_hex(16),owner=seller,delta_vnd=seller_totals[seller],reference="release:"+order_id,created_at=now()))
        record.status="released"
        return {**order_view(record),"is_buyer":True}

@router.post("/orders/{order_id}/cancel")
def cancel(order_id:str,user=Depends(member)):
    with lock,Session.begin() as db:
        record=db.scalar(select(Order).where(Order.id==order_id).with_for_update())
        if not record or record.buyer!=user:
            raise HTTPException(404)
        if record.status=="cancelled":
            return {**order_view(record),"is_buyer":True}
        if record.status!="created":
            raise HTTPException(409,"only_unpaid_order_can_cancel")
        for item in record.details["items"]:
            db.get(Listing,item["listing_id"]).status="active"
        record.status="cancelled"
        return {**order_view(record),"is_buyer":True}

class PersonalInput(BaseModel):
    card_id:str
    variant:str
    kind:Literal["collection","wishlist"]
    quantity:int=Field(default=1,ge=1,le=10000)
    threshold_vnd:int|None=Field(default=None,ge=1,le=2_000_000_000)

@router.post("/personal")
def save_personal(body:PersonalInput,user=Depends(member)):
    with lock,Session.begin() as db:
        card=db.get(Card,body.card_id)
        if not card or body.variant not in {v["id"] for v in catalog_variants(card.details)}:
            raise HTTPException(422,"invalid_card_variant")
        row=db.scalar(select(PersonalCard).where(PersonalCard.owner==user,PersonalCard.card_id==body.card_id,PersonalCard.variant==body.variant,PersonalCard.kind==body.kind))
        if not row:
            row=PersonalCard(id=secrets.token_hex(16),owner=user,**body.model_dump())
            db.add(row)
        else:
            row.quantity=body.quantity;row.threshold_vnd=body.threshold_vnd
        return {"id":row.id,"kind":row.kind}

@router.get("/personal")
def personal(user=Depends(member)):
    from .pricing import quote_payload
    with Session() as db:
        result=[]
        for row in db.scalars(select(PersonalCard).where(PersonalCard.owner==user)):
            quote=quote_payload(db,db.get(Card,row.card_id),row.variant,"NM")
            result.append({"id":row.id,"card_id":row.card_id,"variant":row.variant,"kind":row.kind,"quantity":row.quantity,"threshold_vnd":row.threshold_vnd,"reference_price":quote})
        return result

class ChatInput(BaseModel):
    body:str=Field(min_length=1,max_length=2000)
    reply_to: str | None = None

@router.get("/messages/{listing_id}")
def messages(listing_id:str,user=Depends(member)):
    with Session() as db:
        listing=db.get(Listing,listing_id)
        if not listing:
            raise HTTPException(404)
        if user!=listing.owner and listing.status!="active" and not db.scalar(select(Message.id).where(Message.listing_id==listing_id,Message.buyer==user)):
            raise HTTPException(404)
        statement=select(Message).where(Message.listing_id==listing_id)
        if user!=listing.owner:
            statement=statement.where(Message.buyer==user)
        rows=db.scalars(statement.order_by(Message.created_at).limit(200)).all()
        return [{"id":r.id,"is_mine":r.sender==user,"body":r.body,"created_at":r.created_at,"thread_id":db.scalar(select(Message.id).where(Message.listing_id==listing_id,Message.buyer==r.buyer).order_by(Message.created_at,Message.id).limit(1))} for r in rows]

@router.post("/messages/{listing_id}")
def message(listing_id:str,body:ChatInput,user=Depends(member)):
    with Session.begin() as db:
        listing=db.get(Listing,listing_id)
        if not listing:
            raise HTTPException(404)
        if user==listing.owner:
            replied=db.get(Message,body.reply_to) if body.reply_to else None
            if not replied or replied.listing_id!=listing_id:
                raise HTTPException(422,"reply_to_message_required")
            buyer=replied.buyer
        else:
            if listing.status!="active" and not db.scalar(select(Message.id).where(Message.listing_id==listing_id,Message.buyer==user)):
                raise HTTPException(404)
            buyer=user
        text_body=body.body.strip()
        if not text_body:
            raise HTTPException(422,"empty_message")
        db.add(Message(id=secrets.token_hex(16),listing_id=listing_id,buyer=buyer,sender=user,body=text_body,created_at=now()))
        return {"status":"sent"}

class AlertInput(BaseModel):
    card_id:str
    variant:str
    direction:Literal["below","above"]
    threshold_vnd:int=Field(ge=1,le=2_000_000_000)

@router.post("/alerts")
def add_alert(body:AlertInput,user=Depends(member)):
    with Session.begin() as db:
        card=db.get(Card,body.card_id)
        if not card or body.variant not in {v["id"] for v in catalog_variants(card.details)}:
            raise HTTPException(422,"invalid_card_variant")
        row=PriceAlert(id=secrets.token_hex(16),owner=user,card_id=body.card_id,variant=body.variant,direction=body.direction,threshold_vnd=body.threshold_vnd,active=True,triggered_at=None)
        db.add(row)
        return {"id":row.id,"status":"armed"}

@router.get("/alerts")
def alerts(user=Depends(member)):
    with Session() as db:
        return [{"id":a.id,"card_id":a.card_id,"variant":a.variant,"direction":a.direction,"threshold_vnd":a.threshold_vnd,"active":a.active,"triggered_at":a.triggered_at} for a in db.scalars(select(PriceAlert).where(PriceAlert.owner==user).order_by(PriceAlert.id.desc()))]

@router.get("/notifications")
def notifications(user=Depends(member)):
    with Session() as db:
        rows=db.scalars(select(Notification).where(Notification.owner==user).order_by(Notification.created_at.desc()).limit(100)).all()
        return [{"id":n.id,"created_at":n.created_at,"alert_id":n.alert_id,"payload":n.payload} for n in rows]

from .db import PriceAlert, Notification


def expire_reservations(db):
    cutoff=(datetime.now(timezone.utc)-timedelta(minutes=15)).isoformat()
    expired=db.scalars(select(Order).where(Order.status=="created",Order.created_at<cutoff).with_for_update()).all()
    for order in expired:
        for item in order.details["items"]:
            db.execute(update(Listing).where(Listing.id==item["listing_id"],Listing.status=="reserved").values(status="active"))
        order.status="cancelled"
    return len(expired)
