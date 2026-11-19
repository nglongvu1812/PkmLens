import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from .db import Session, Identity, User, Login, LoginAttempt
from .catalog import now

router=APIRouter(prefix="/api/v1")

def extract_token(request: Request) -> str:
    token = request.cookies.get("pkmlens_session", "")
    if not token:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
        elif request.headers.get("x-session-token"):
            token = request.headers.get("x-session-token", "")
        elif hasattr(request, "query_params") and request.query_params.get("token"):
            token = request.query_params.get("token", "")
    return token

def owner(request: Request):
    token = extract_token(request)
    if not token:
        raise HTTPException(401, "session_required")
    digest = hashlib.sha256(token.encode()).hexdigest()
    with Session() as db:
        identity = db.get(Identity, digest)
        if not identity or identity.expires_at < now():
            raise HTTPException(401, "session_expired")
        login = db.get(Login, digest)
        return db.get(User, login.user_id).owner if login else digest

def member(user=Depends(owner)):
    with Session() as db:
        if not db.scalar(select(User).where(User.owner==user)):
            raise HTTPException(401,"account_required")
    return user

def password_hash(password, salt=None):
    salt=salt or secrets.token_hex(16)
    digest=hashlib.scrypt(password.encode(),salt=bytes.fromhex(salt),n=16384,r=8,p=1,maxmem=64*1024*1024).hex()
    return salt+":"+digest

class Credentials(BaseModel):
    email: str = Field(min_length=3,max_length=254)
    password: str = Field(min_length=12,max_length=128)
    display_name: str = Field(default="Collector",min_length=1,max_length=80)

def email_key(email):
    email=email.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+",email):
        raise HTTPException(422,"invalid_email")
    return email

@router.post("/auth/register")
def register(body: Credentials,request: Request,response: Response,user=Depends(owner)):
    try:
        with Session.begin() as db:
            if db.scalar(select(User).where(User.owner==user)):
                raise HTTPException(409,"already_registered")
            record=User(id=secrets.token_hex(16),email=email_key(body.email),password=password_hash(body.password),owner=user,display_name=body.display_name.strip())
            db.add(record)
            db.flush()
            guest=db.get(Identity,user)
            if guest:
                guest.expires_at=now()
            token=secrets.token_urlsafe(32)
            digest=hashlib.sha256(token.encode()).hexdigest()
            db.add(Identity(id=digest,created_at=now(),expires_at=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()))
            db.flush()
            db.add(Login(id=digest,user_id=record.id))
        import os
        response.set_cookie("pkmlens_session",token,httponly=True,samesite="strict",secure=os.getenv("COOKIE_SECURE","false")=="true",max_age=604800)
        return {"display_name":body.display_name,"mode":"simulated_market"}
    except IntegrityError as exc:
        raise HTTPException(409,"email_or_session_already_registered") from exc

@router.post("/auth/login")
def login(body: Credentials,request: Request,response: Response):
    address=request.client.host if request.client else "unknown"
    attempt_id=hashlib.sha256((email_key(body.email)+"|"+address).encode()).hexdigest()
    failure=False
    result=None
    token=None
    with Session.begin() as db:
        attempt=db.get(LoginAttempt,attempt_id,with_for_update=True)
        if attempt and attempt.blocked_until and attempt.blocked_until>now():
            raise HTTPException(429,"login_rate_limited")
        if attempt is None:
            attempt=LoginAttempt(id=attempt_id,failed_count=0,window_started=now())
            db.add(attempt)
        elif datetime.fromisoformat(attempt.window_started)<datetime.now(timezone.utc)-timedelta(minutes=15):
            attempt.failed_count=0
            attempt.window_started=now()
            attempt.blocked_until=None
        user=db.scalar(select(User).where(User.email==email_key(body.email)))
        stored=user.password if user else password_hash("not-a-valid-account-password")
        salt,_=stored.split(":")
        if not hmac.compare_digest(stored,password_hash(body.password,salt)) or not user:
            attempt.failed_count+=1
            if attempt.failed_count>=8:
                attempt.blocked_until=(datetime.now(timezone.utc)+timedelta(minutes=15)).isoformat()
            failure=True
        else:
            attempt.failed_count=0
            attempt.blocked_until=None
            token=secrets.token_urlsafe(32)
            digest=hashlib.sha256(token.encode()).hexdigest()
            db.add(Identity(id=digest,created_at=now(),expires_at=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()))
            db.flush()
            db.add(Login(id=digest,user_id=user.id))
            result={"display_name":user.display_name,"mode":"simulated_market"}
    if failure:
        raise HTTPException(401,"invalid_credentials")
    import os
    response.set_cookie("pkmlens_session",token,httponly=True,samesite="strict",secure=os.getenv("COOKIE_SECURE","false")=="true",max_age=604800)
    return result
@router.get("/me")
def me(user=Depends(owner)):
    with Session() as db:
        record=db.scalar(select(User).where(User.owner==user))
        from .db import RoleGrant
        grant=db.get(RoleGrant,user)
        return {"display_name":record.display_name if record else None,"registered":record is not None,"owner":user,"role":grant.role if grant else "member" if record else "guest"}

@router.post("/auth/logout")
def logout(request: Request,response: Response):
    token=request.cookies.get("pkmlens_session","")
    digest=hashlib.sha256(token.encode()).hexdigest()
    with Session.begin() as db:
        record=db.get(Login,digest)
        if record:
            db.delete(record)
        identity=db.get(Identity,digest)
        if identity:
            identity.expires_at=now()
    response.delete_cookie("pkmlens_session")
    return {"status":"logged_out"}



def moderator(user=Depends(member)):
    from .db import RoleGrant
    with Session() as db:
        grant=db.get(RoleGrant,user)
        if not grant or grant.role not in ("admin","moderator"):
            raise HTTPException(403,"moderator_required")
    return user
