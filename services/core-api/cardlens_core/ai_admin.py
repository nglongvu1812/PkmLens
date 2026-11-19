"""Role-scoped gateway to local AI training and checkpoint management."""
import logging
import os
import secrets
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import httpx
from .accounts import moderator
from .catalog import now
from .db import Audit, Session

router=APIRouter(prefix='/api/v1/admin/ai')
log=logging.getLogger(__name__)


async def call(path,method='GET',body=None):
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            result=await client.request(method,os.getenv('AI_URL','http://127.0.0.1:8101')+'/v1/training'+path,
                headers={'x-cardlens-key':os.getenv('AI_INTERNAL_KEY','')},json=body)
        if result.status_code>=400:raise HTTPException(result.status_code,result.json().get('detail','ai_operation_failed'))
        return result.json()
    except httpx.HTTPError as exc:
        log.warning('ai_admin_unavailable path=%s error=%s',path,type(exc).__name__)
        raise HTTPException(503,'ai_service_unavailable') from exc


def audit(user,action,target):
    with Session.begin() as db:
        db.add(Audit(id=secrets.token_hex(16),actor=user,action='ai_'+action,target=target,created_at=now(),details={}))


@router.get('/status')
async def status(user=Depends(moderator)):
    return await call('/status')


@router.post('/datasets/{dataset_id}/validate')
async def validate(dataset_id:str,user=Depends(moderator)):
    if not dataset_id.replace('-','').replace('_','').isalnum():raise HTTPException(422)
    return await call('/datasets/'+dataset_id+'/validate','POST')


class Train(BaseModel):
    dataset:str=Field(pattern=r'^[A-Za-z0-9_-]{1,80}$')
    model:Literal['n','s','m']='n'
    epochs:int=Field(25,ge=1,le=300)
    batch:int=Field(4,ge=1,le=16)
    imgsz:Literal[320,480,640]=480
    device:Literal['auto','cpu','0']='auto'
    resume_run:str|None=None
    checkpoint:str|None=None


@router.post('/runs')
async def train(body:Train,user=Depends(moderator)):
    result=await call('/runs','POST',body.model_dump());audit(user,'train',result['id']);return result


class Activation(BaseModel):
    experimental:bool=False


@router.post('/runs/{run_id}/{action}')
async def action(run_id:str,action:Literal['stop','activate','export'],body:Activation,user=Depends(moderator)):
    if not run_id.replace('-','').replace('_','').isalnum():raise HTTPException(422)
    result=await call('/runs/'+run_id+'/'+action,'POST',body.model_dump())
    audit(user,action,run_id);return result
