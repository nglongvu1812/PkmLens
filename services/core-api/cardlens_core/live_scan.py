"""Authenticated scan gateway; the AI service never owns prices or user permissions."""
import asyncio
import json
import logging
import os
import time
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
import httpx
from .accounts import owner, extract_token
from .db import Session, Card
from .shop import card_summary

router=APIRouter(prefix='/api/v1')
log=logging.getLogger(__name__)


@router.get('/live/status')
async def status(user=Depends(owner)):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response=await client.get(os.getenv('AI_URL','http://127.0.0.1:8101')+'/v1/detectors',
                                      headers={'x-cardlens-key':os.getenv('AI_INTERNAL_KEY','')})
            response.raise_for_status();data=response.json()
            return {key:data[key] for key in ('available','active','models','pose_active','pose_models','cuda','device','gallery') if key in data}
    except (httpx.HTTPError,ValueError):
        return {'available':False,'models':[],'active':None,'reason':'ai_unavailable'}


@router.websocket('/ws/scan')
async def scan(ws:WebSocket):
    origin = ws.headers.get('origin')
    allowed = [x.strip() for x in os.getenv('WEB_ORIGIN', 'http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:3000,http://localhost:3000,*').split(',') if x.strip()]
    is_allowed = '*' in allowed or not origin or origin in allowed or origin.endswith('.vercel.app') or origin.endswith('.loca.lt') or origin.endswith('.ngrok-free.app')
    if not is_allowed:
        await ws.close(code=1008)
        return
    try:
        user = owner(ws)
    except HTTPException:
        user = "guest" 
    key=os.getenv('AI_INTERNAL_KEY','')
    if not key:
        await ws.close(code=1011);return
    from websockets.asyncio.client import connect
    address=os.getenv('AI_URL','http://127.0.0.1:8101').replace('http://','ws://').replace('https://','wss://')+'/v1/live'
    await ws.accept()
    workers=[]
    try:
        async with connect(address,additional_headers={'x-cardlens-key':key},max_size=3_000_000,max_queue=2) as ai:
            async def upload():
                last=0
                while True:
                    message=await ws.receive()
                    if message['type']=='websocket.disconnect':return
                    if message.get('bytes') is not None:
                        raw=message['bytes']
                        if len(raw)>1_500_000:await ws.close(code=1009);return
                        now=time.monotonic()
                        if now-last<.08:continue
                        last=now;await ai.send(raw)
                    elif message.get('text'):
                        if len(message['text'])>2048:await ws.close(code=1009);return
                        await ai.send(message['text'])
            async def results():
                async for raw in ai:
                    value=json.loads(raw)
                    with Session() as db:
                        candidates=value.get('candidates',[])+[c for t in value.get('tracks',[]) for c in t.get('candidates',[])]
                        for candidate in candidates:
                            card=db.get(Card,candidate['card_id'])
                            candidate['card']=card_summary(card) if card else None
                    await ws.send_json(value)
            workers=[asyncio.create_task(upload()),asyncio.create_task(results())]
            done,_=await asyncio.wait(workers,return_when=asyncio.FIRST_COMPLETED)
            for task in done:task.result()
    except (HTTPException,WebSocketDisconnect):pass
    except Exception:
        log.exception('live_gateway_failed')
        try:await ws.send_json({'type':'error','detail':'live_service_unavailable'})
        except Exception:pass
    finally:
        for task in workers:task.cancel()
        await asyncio.gather(*workers,return_exceptions=True)
        try:await ws.close()
        except (RuntimeError, WebSocketDisconnect):pass
