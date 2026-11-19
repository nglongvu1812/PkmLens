import hashlib
import json
import os
import re
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Literal
import httpx
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, text, func
from sqlalchemy.exc import IntegrityError
from fastapi.middleware.cors import CORSMiddleware
from .db import Session, Card, Scan, Quote, Listing, Identity, DATA, URL
from .catalog import catalog_variants, now, variants
from .catalog_images import catalog_image
from .pricing import quote_payload

app = FastAPI(title="PkmLens API", version="0.1.0")
mutation_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def boundary(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        allowed = os.getenv("WEB_ORIGIN", "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:3000,http://localhost:3000,*").split(",")
        allowed_list = [x.strip() for x in allowed if x.strip()]
        is_allowed = "*" in allowed_list or not origin or origin in allowed_list or origin.endswith(".vercel.app") or origin.endswith(".loca.lt") or origin.endswith(".ngrok-free.app")
        if not is_allowed:
            return JSONResponse({"detail": "origin_not_allowed"}, status_code=403)
        length = request.headers.get("content-length")
        maximum=17*1024*1024 if request.url.path.endswith("/photos") else 9*1024*1024
        if length and (not length.isdigit() or int(length) > maximum):
            return JSONResponse({"detail": "request_too_large"}, status_code=413)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

from .accounts import owner

def get_owned(db, model, ident, user):
    row = db.get(model, ident)
    if not row or row.owner != user:
        raise HTTPException(404, "not_found")
    return row

class Health(BaseModel):
    status: str
    database: str
    ai: dict

@app.get("/healthz")
def health():
    return {"status": "ok", "service": "core-api"}

@app.get("/api/v1/health", response_model=Health)
def readiness(response: Response):
    try:
        with Session() as db:
            db.execute(text("SELECT 1"))
            count = db.scalar(select(func.count()).select_from(Card))
    except Exception:
        response.status_code = 503
        return Health(status="not_ready", database="unavailable", ai={})
    try:
        result = httpx.get(os.getenv("AI_URL", "http://127.0.0.1:8101") + "/healthz", timeout=3)
        result.raise_for_status()
        ai = result.json()
    except (httpx.HTTPError, ValueError):
        ai = {"status": "unavailable", "pose_ready": False, "identification_ready": False}
    return Health(status="ready" if ai.get("identification_ready") and count else "degraded", database="sqlite_local_fallback" if URL.startswith("sqlite") else "postgresql", ai={**ai, "catalog_count": count})

@app.get("/readyz")
def ready(response: Response):
    result = readiness(response)
    if result.status != "ready":
        response.status_code = 503
    return result

@app.post("/api/v1/session")
def create_session(request: Request, response: Response):
    try:
        digest = owner(request)
        return {"mode": "local_guest", "expires_days": 7}
    except HTTPException:
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).hexdigest()
        with Session.begin() as db:
            db.add(Identity(id=digest, created_at=now(), expires_at=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()))
        is_https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https" or os.getenv("COOKIE_SECURE", "false") == "true"
        response.set_cookie("pkmlens_session", token, httponly=True, samesite="none" if is_https else "lax", secure=is_https, max_age=604800)
        return {"mode": "local_guest", "expires_days": 7, "token": token}

class CardView(BaseModel):
    id: str
    provider_id: str
    set_id: str
    number: str
    language: str
    name: str
    image: str
    variants: list[dict]

def card_view(card):
    return CardView(id=card.id, provider_id=card.provider_id, set_id=card.set_id, number=card.number, language=card.language, name=card.name, image="/api/v1/cards/"+card.id+"/image", variants=[{k: v for k,v in variant.items() if k != "raw"} for variant in catalog_variants(card.details)])

@app.get("/api/v1/cards", response_model=list[CardView])
def cards(q: str = "", set_id: str | None = None):
    with Session() as db:
        statement = select(Card)
        if q:
            statement = statement.where((Card.name.ilike("%"+q[:100]+"%")) | (Card.number == q[:40]))
        if set_id:
            statement = statement.where(Card.set_id == set_id)
        return [card_view(c) for c in db.scalars(statement.order_by(Card.set_id, Card.number).limit(300))]

@app.get("/api/v1/cards/{card_id}/image")
def card_image(card_id: str):
    with Session() as db:
        card = db.get(Card, card_id)
        if card is None:
            raise HTTPException(404)
        provider_id, details = card.provider_id, card.details
    path = None
    if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", provider_id):
        gallery_root = (DATA / "gallery").resolve()
        gallery = gallery_root / (provider_id + ".webp")
        if gallery.is_file() and gallery.resolve().is_relative_to(gallery_root):
            path = gallery
    if path is None:
        path = catalog_image(details, DATA)
    if path is None:
        raise HTTPException(404)
    return FileResponse(path, media_type="image/webp")

@app.get("/api/v1/cards/{card_id}", response_model=CardView)
def card_detail(card_id: str):
    with Session() as db:
        card = db.get(Card, card_id)
        if not card:
            raise HTTPException(404)
        return card_view(card)

class Candidate(BaseModel):
    card_id: str
    similarity: float
    fusion_score: float | None = None
    evidence: dict = Field(default_factory=dict)
    card: CardView | None = None

class ScanView(BaseModel):
    scan_id: str
    status: str
    candidates: list[Candidate]
    before_ocr: list[Candidate]
    ocr: dict
    warped: str
    corners: list[list[float]]
    mode: str
    reason_codes: list[str]
    model_versions: dict
    timing_ms: dict

@app.post("/api/v1/scan/identify", response_model=ScanView)
def identify(image: UploadFile = File(), mode: Literal["cropped", "manual_corners"] = Form("cropped"), corners: str | None = Form(None), user: str = Depends(owner)):
    from cardlens_ai.imaging import decode, MAX_BYTES
    try:
        raw = image.file.read(MAX_BYTES+1)
        clean = decode(raw)
        import io
        out = io.BytesIO()
        clean.save(out, "WEBP", quality=95)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with Session() as db:
        recent = (datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
        if db.scalar(select(func.count()).select_from(Scan).where(Scan.owner == user, Scan.created_at > recent)) >= 10:
            raise HTTPException(429, "scan_rate_limit")
    try:
        data = {"mode": mode}
        if corners:
            data["corners"] = corners
        result = httpx.post(os.getenv("AI_URL", "http://127.0.0.1:8101") + "/v1/identify", files={"image": ("scan.webp", out.getvalue(), "image/webp")}, data=data, timeout=120)
        if result.status_code != 200:
            raise HTTPException(result.status_code if result.status_code in (422, 503) else 502, "ai_identification_failed")
        value = result.json()
    except httpx.HTTPError as exc:
        raise HTTPException(503, "ai_unavailable") from exc
    with Session.begin() as db:
        for candidate in value["candidates"] + value["before_ocr"]:
            card = db.get(Card, candidate["card_id"])
            candidate["card"] = card_view(card).model_dump() if card else None
        ident = secrets.token_hex(16)
        # No user image bytes are persisted; evidence previews exist in this response only.
        stored = {k: v for k, v in value.items() if k != "warped"}
        stored["ocr"] = {k: v for k,v in value["ocr"].items() if k != "crop"}
        db.add(Scan(id=ident, owner=user, created_at=now(), result=stored, condition="NM", revision=0))
    return ScanView(scan_id=ident, **value)

class Confirmation(BaseModel):
    card_id: str
    variant: str
    condition: Literal["NM", "LP", "MP", "HP", "DMG"] = "NM"

class QuoteView(BaseModel):
    quote_id: str
    scan_id: str
    revision: int
    price: dict

@app.post("/api/v1/scan/{scan_id}/confirm", response_model=QuoteView)
def confirm(scan_id: str, body: Confirmation, user: str = Depends(owner)):
    with mutation_lock, Session.begin() as db:
        scan = db.scalar(select(Scan).where(Scan.id == scan_id).with_for_update())
        if not scan or scan.owner != user:
            raise HTTPException(404)
        card = db.get(Card, body.card_id)
        if not card or body.variant not in {v["id"] for v in catalog_variants(card.details)}:
            raise HTTPException(422, "invalid_card_variant")
        if db.scalar(select(Listing).where(Listing.scan_id == scan_id)):
            raise HTTPException(409, "draft_already_created_start_new_scan")
        scan.card_id, scan.variant, scan.condition = body.card_id, body.variant, body.condition
        scan.revision += 1
        payload = quote_payload(db, card, body.variant, body.condition)
        ident = secrets.token_hex(16)
        db.add(Quote(id=ident, scan_id=scan.id, owner=user, revision=scan.revision, created_at=now(), payload=payload))
        return QuoteView(quote_id=ident, scan_id=scan.id, revision=scan.revision, price=payload)

class DraftInput(BaseModel):
    quote_id: str
    asking_vnd: int | None = Field(default=None, ge=1, le=2_000_000_000)

class DraftView(BaseModel):
    id: str
    card_id: str
    variant: str
    condition: str
    quote_id: str
    asking_vnd: int | None
    status: str
    needs_photos: bool

@app.post("/api/v1/listings/from-scan/{scan_id}", response_model=DraftView)
def create_draft(scan_id: str, body: DraftInput, user: str = Depends(owner)):
    with mutation_lock, Session.begin() as db:
        scan = db.scalar(select(Scan).where(Scan.id == scan_id).with_for_update())
        if not scan or scan.owner != user:
            raise HTTPException(404)
        quote = get_owned(db, Quote, body.quote_id, user)
        if quote.scan_id != scan_id or quote.revision != scan.revision:
            raise HTTPException(409, "stale_quote_reconfirm")
        if quote.created_at < (datetime.now(timezone.utc)-timedelta(minutes=30)).isoformat():
            raise HTTPException(409, "quote_expired_reconfirm")
        listing = db.scalar(select(Listing).where(Listing.scan_id == scan_id))
        if listing is None:
            listing = Listing(id=secrets.token_hex(16), owner=user, scan_id=scan_id, quote_id=quote.id, card_id=scan.card_id, variant=scan.variant, condition=scan.condition, asking_vnd=body.asking_vnd, created_at=now(), status="draft", details={"training_consent": False})
            db.add(listing)
            db.flush()
        return DraftView(id=listing.id, card_id=listing.card_id, variant=listing.variant, condition=listing.condition, quote_id=listing.quote_id, asking_vnd=listing.asking_vnd, status=listing.status, needs_photos=True)

@app.get("/api/v1/listings/drafts", response_model=list[DraftView])
def drafts(user: str = Depends(owner)):
    with Session() as db:
        return [DraftView(id=r.id, card_id=r.card_id, variant=r.variant, condition=r.condition, quote_id=r.quote_id, asking_vnd=r.asking_vnd, status=r.status, needs_photos=True) for r in db.scalars(select(Listing).where(Listing.owner == user, Listing.status == "draft").order_by(Listing.created_at.desc()))]


@app.get("/api/v1/experiments")
def experiments(user: str = Depends(owner)):
    # shortcut: local read-only reports; require admin role when account/RBAC ships.
    runs = []
    for path in sorted((DATA / "benchmarks").glob("*/summary.json")):
        try:
            runs.append({"run_id": path.parent.name, "summary": json.loads(path.read_text(encoding="utf-8"))})
        except (ValueError, OSError):
            continue
    return runs

@app.get("/api/v1/experiments/{run_id}/csv")
def experiment_csv(run_id: str, user: str = Depends(owner)):
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(422, "invalid_run_id")
    path = DATA / "benchmarks" / run_id / "results.csv"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="text/csv", filename=run_id+".csv")


from .limits import BodyLimit
app.add_middleware(BodyLimit)


from .accounts import router as accounts_router
from .market import router as market_router
app.include_router(accounts_router)
app.include_router(market_router)

@app.get("/api/v1/cards/{card_id}/prices")
def price_history(card_id:str,variant:str,user: str = Depends(owner)):
    from .db import Price
    with Session() as db:
        card=db.get(Card,card_id)
        if not card or variant not in {v["id"] for v in catalog_variants(card.details)}:
            raise HTTPException(404,"card_variant_not_found")
        from .db import Mapping
        rows=db.scalars(select(Price).where(Price.card_id==card_id,Price.variant==variant).order_by(Price.source_at.desc()).limit(90)).all()
        rows=[p for p in rows if db.scalar(select(Mapping).where(Mapping.card_id==p.card_id,Mapping.variant==p.variant,Mapping.provider==p.source.split("/")[-1],Mapping.status=="matched"))]
        return [{"id":p.id,"source":p.source,"source_at":p.source_at,"fetched_at":p.fetched_at,"currency":p.currency,"amount":p.amount,"low":p.low,"high":p.high} for p in rows]


from .extensions import router as extensions_router
app.include_router(extensions_router)

from .realtime import router as realtime_router
app.include_router(realtime_router)

from .shop import router as shop_router
app.include_router(shop_router)

from .live_scan import router as live_scan_router
app.include_router(live_scan_router)

from .ai_admin import router as ai_admin_router
app.include_router(ai_admin_router)
