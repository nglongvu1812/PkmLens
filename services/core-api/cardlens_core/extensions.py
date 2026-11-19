import json
import secrets
from datetime import datetime,timezone
from pathlib import Path
from fastapi import APIRouter,Depends,HTTPException
from pydantic import BaseModel,Field
from sqlalchemy import select
from .accounts import member,moderator,owner
from .catalog import catalog_variants,now,variants,key
from .db import Session,Card,Listing,Scan,Quote,PersonalCard,CollectionValue,Audit,PriceAlert,Message
from .pricing import quote_payload

router=APIRouter(prefix="/api/v1")

class ManualDraft(BaseModel):
    card_id:str
    variant:str
    condition:str=Field(pattern="^(NM|LP|MP|HP|DMG)$",default="NM")
    asking_vnd:int|None=Field(default=None,ge=1,le=2_000_000_000)

@router.post("/listings/manual")
def manual_draft(body:ManualDraft,user=Depends(member)):
    with Session.begin() as db:
        card=db.get(Card,body.card_id)
        if not card or body.variant not in {v["id"] for v in catalog_variants(card.details)}:
            raise HTTPException(422,"invalid_card_variant")
        scan_id,quote_id,listing_id=[secrets.token_hex(16) for _ in range(3)]
        db.add(Scan(id=scan_id,owner=user,created_at=now(),result={"mode":"manual_catalog_selection","not_ai_identified":True},card_id=card.id,variant=body.variant,condition=body.condition,revision=1))
        db.flush()
        db.add(Quote(id=quote_id,scan_id=scan_id,owner=user,revision=1,created_at=now(),payload=quote_payload(db,card,body.variant,body.condition)))
        db.flush()
        db.add(Listing(id=listing_id,owner=user,scan_id=scan_id,quote_id=quote_id,card_id=card.id,variant=body.variant,condition=body.condition,asking_vnd=body.asking_vnd,status="draft",created_at=now(),details={"training_consent":False,"source":"manual"}))
        return {"id":listing_id,"status":"draft"}

@router.delete("/personal/{item_id}")
def delete_personal(item_id:str,user=Depends(member)):
    with Session.begin() as db:
        row=db.get(PersonalCard,item_id)
        if not row or row.owner!=user:raise HTTPException(404)
        db.delete(row)
    return {"status":"deleted"}

@router.get("/collection/value")
def collection_value(user=Depends(member)):
    with Session.begin() as db:
        total,missing=0,0
        for item in db.scalars(select(PersonalCard).where(PersonalCard.owner==user,PersonalCard.kind=="collection")):
            price=quote_payload(db,db.get(Card,item.card_id),item.variant,"NM")["amount_vnd"]
            if price is None:missing+=item.quantity
            else:total+=price*item.quantity
        day=datetime.now(timezone.utc).date().isoformat()
        db.merge(CollectionValue(id=key(user+day),owner=user,day=day,amount_vnd=str(total),missing_count=missing,created_at=now()))
        db.flush()
        history=db.scalars(select(CollectionValue).where(CollectionValue.owner==user).order_by(CollectionValue.day)).all()
        return {"known_value_vnd":total,"missing_price_quantity":missing,"condition_assumption":"NM","history":[{"day":r.day,"amount_vnd":int(r.amount_vnd),"missing_count":r.missing_count} for r in history]}

@router.delete("/alerts/{alert_id}")
def cancel_alert(alert_id:str,user=Depends(member)):
    with Session.begin() as db:
        row=db.get(PriceAlert,alert_id)
        if not row or row.owner!=user:raise HTTPException(404)
        row.active=False
    return {"status":"disabled"}

@router.get("/inbox")
def inbox(user=Depends(member)):
    with Session() as db:
        rows=db.execute(select(Message,Listing).join(Listing,Message.listing_id==Listing.id).where((Message.buyer==user)|(Listing.owner==user)).order_by(Message.created_at)).all()
        threads={}
        for message,listing in rows:
            pair=(listing.id,message.buyer)
            if pair not in threads:threads[pair]={"thread_id":message.id,"listing_id":listing.id,"card_id":listing.card_id,"is_seller":listing.owner==user}
            threads[pair].update(last_message=message.body,updated_at=message.created_at)
        return list(threads.values())

@router.get("/admin/moderation")
def moderation_queue(user=Depends(moderator)):
    with Session() as db:
        rows=db.scalars(select(Listing).where(Listing.status=="pending_review").order_by(Listing.created_at)).all()
        return [{"id":r.id,"card_id":r.card_id,"variant":r.variant,"details":r.details,"image":"/api/v1/listings/"+r.id+"/photos/front"} for r in rows]

class Decision(BaseModel):
    action:str=Field(pattern="^(approve|reject)$")
    reason:str=Field(min_length=5,max_length=1000)

@router.post("/admin/moderation/{listing_id}")
def moderation_decision(listing_id:str,body:Decision,user=Depends(moderator)):
    with Session.begin() as db:
        row=db.get(Listing,listing_id,with_for_update=True)
        if not row or row.status!="pending_review":raise HTTPException(409,"listing_not_pending")
        if row.owner==user:raise HTTPException(403,"cannot_review_own_listing")
        row.status="active" if body.action=="approve" else "removed"
        row.details={**row.details,"moderation_decision":{"actor":user,"reason":body.reason,"at":now(),"action":body.action}}
        db.add(Audit(id=secrets.token_hex(16),actor=user,action="listing_"+body.action,target=row.id,created_at=now(),details={"reason":body.reason}))
        return {"id":row.id,"status":row.status}

@router.get("/admin/audit")
def audit(user=Depends(moderator)):
    with Session() as db:
        return [{"id":r.id,"action":r.action,"target":r.target,"created_at":r.created_at,"details":r.details} for r in db.scalars(select(Audit).order_by(Audit.created_at.desc()).limit(100))]


@router.get("/experiments/{run_id}/samples")
def experiment_samples(run_id:str,method:str="full",incorrect_only:bool=False,user=Depends(owner)):
    from .db import DATA
    if not run_id.replace("-","").replace("_","").isalnum():raise HTTPException(422)
    if method not in ("full","phash","embedding","embedding_ocr"):raise HTTPException(422)
    path=DATA/"benchmarks"/run_id/"results.jsonl"
    if not path.is_file():raise HTTPException(404)
    result=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        if row.get("method")!=method:continue
        correct=row.get("ranked_ids",[None])[0]==row.get("ground_truth") if row.get("ground_truth") else row.get("status")=="unknown"
        if incorrect_only and correct:continue
        result.append(row)
        if len(result)>=1000:break
    return result


class Shipping(BaseModel):
    simulated_tracking:str=Field(min_length=3,max_length=100)

@router.post("/orders/{order_id}/ship")
def ship(order_id:str,body:Shipping,user=Depends(member)):
    from .db import Order
    with Session.begin() as db:
        row=db.get(Order,order_id,with_for_update=True)
        if not row or user not in {i["seller"] for i in row.details["items"]}:raise HTTPException(404)
        if row.status not in ("paid_escrow","shipped"):raise HTTPException(409,"order_not_paid")
        shipments={**row.details.get("shipments",{}),user:{"tracking":body.simulated_tracking,"at":now(),"simulated":True}}
        row.details={**row.details,"shipments":shipments}
        if all(i["seller"] in shipments for i in row.details["items"]):row.status="shipped"
        return {"id":row.id,"status":row.status,"simulated":True}

class Dispute(BaseModel):
    reason:str=Field(min_length=10,max_length=2000)

@router.post("/orders/{order_id}/dispute")
def dispute(order_id:str,body:Dispute,user=Depends(member)):
    from .db import Order
    with Session.begin() as db:
        row=db.get(Order,order_id,with_for_update=True)
        if not row or row.buyer!=user:raise HTTPException(404)
        if row.status=="disputed":return {"id":row.id,"status":row.status}
        if row.status not in ("paid_escrow","shipped"):raise HTTPException(409,"invalid_dispute_state")
        row.details={**row.details,"dispute":{"reason":body.reason,"at":now()}}
        row.status="disputed"
        db.add(Audit(id=secrets.token_hex(16),actor=user,action="order_disputed",target=row.id,created_at=now(),details={"reason":body.reason}))
        return {"id":row.id,"status":row.status}

class Resolution(BaseModel):
    action:str=Field(pattern="^(refund|release)$")
    reason:str=Field(min_length=10,max_length=2000)

@router.get("/admin/disputes")
def disputes(user=Depends(moderator)):
    from .db import Order
    with Session() as db:
        return [{"id":r.id,"total_vnd":r.total_vnd,"dispute":r.details.get("dispute")} for r in db.scalars(select(Order).where(Order.status=="disputed"))]

@router.post("/admin/disputes/{order_id}/resolve")
def resolve_dispute(order_id:str,body:Resolution,user=Depends(moderator)):
    from .db import Order,Ledger,User
    from .market import lock
    with lock,Session.begin() as db:
        row=db.get(Order,order_id,with_for_update=True)
        if not row:raise HTTPException(404)
        participants={row.buyer}|{i["seller"] for i in row.details["items"]}
        if user in participants:raise HTTPException(403,"independent_reviewer_required")
        resolution=row.details.get("resolution")
        if resolution:
            if resolution["action"]!=body.action:raise HTTPException(409,"resolution_already_final")
            return {"id":row.id,"status":row.status,"simulated":True}
        if row.status!="disputed":raise HTTPException(409,"order_not_disputed")
        totals={}
        if body.action=="refund":totals[row.buyer]=row.total_vnd
        else:
            for item in row.details["items"]:totals[item["seller"]]=totals.get(item["seller"],0)+item["price_vnd"]
        for recipient in sorted(totals):
            db.scalar(select(User).where(User.owner==recipient).with_for_update())
            db.add(Ledger(id=secrets.token_hex(16),owner=recipient,delta_vnd=totals[recipient],reference="resolution:"+row.id,created_at=now()))
        for item in row.details["items"]:db.get(Listing,item["listing_id"]).status="removed" if body.action=="refund" else "sold"
        row.status="refunded" if body.action=="refund" else "released"
        row.details={**row.details,"resolution":{"action":body.action,"reason":body.reason,"actor":user,"at":now()}}
        db.add(Audit(id=secrets.token_hex(16),actor=user,action="dispute_"+body.action,target=row.id,created_at=now(),details={"reason":body.reason}))
        return {"id":row.id,"status":row.status,"simulated":True}
