"""Owner-scoped local WebSocket updates. DB polling also works across API processes."""
import asyncio
from fastapi import APIRouter,WebSocket,WebSocketDisconnect,HTTPException
from sqlalchemy import select
from .accounts import owner
from .db import Session,Listing,Message,Identity
router=APIRouter()

@router.websocket("/api/v1/ws/chat/{listing_id}/{thread_id}")
async def chat_events(ws:WebSocket,listing_id:str,thread_id:str):
    origins={"http://127.0.0.1:3000","http://localhost:3000"}
    if ws.headers.get("origin") not in origins:
        await ws.close(code=1008);return
    try:
        user=owner(ws)
    except HTTPException:
        await ws.close(code=1008);return
    with Session() as db:
        listing=db.get(Listing,listing_id)
        first=db.get(Message,thread_id)
        if not listing or not first or first.listing_id!=listing_id or user not in (listing.owner,first.buyer):
            await ws.close(code=1008);return
        buyer=first.buyer
    await ws.accept()
    last=None
    try:
        while True:
            try:current=owner(ws)
            except HTTPException:
                await ws.close(code=1008);return
            if current!=user:
                await ws.close(code=1008);return
            with Session() as db:
                rows=db.scalars(select(Message).where(Message.listing_id==listing_id,Message.buyer==buyer).order_by(Message.created_at,Message.id).limit(200)).all()
                payload=[{"id":m.id,"body":m.body,"is_mine":m.sender==user,"thread_id":thread_id,"created_at":m.created_at} for m in rows]
            if payload!=last:
                await ws.send_json({"type":"chat.messages","messages":payload})
                last=payload
            try:
                event=await asyncio.wait_for(ws.receive_text(),timeout=1)
                if event!="ping":await ws.send_json({"type":"error","detail":"use_authorized_http_message_endpoint"})
            except asyncio.TimeoutError:pass
    except WebSocketDisconnect:
        return
