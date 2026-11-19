"""Storefront reads and explicit variant quotes over the existing business tables."""
import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, case
from .accounts import owner
from .catalog import catalog_variants, now, variants
from .catalog_images import picture_id
from .db import DATA, Session, Card, Listing, PersonalCard, Quote, Scan, User
from .live_price import fetch_live_price_for_card
from .pricing import quote_payload

router = APIRouter(prefix="/api/v1")


def card_summary(card, db=None):
    options = variants(card.details)
    result = {"id": card.id, "name": card.name, "number": card.number,
            "set_id": card.set_id, "set_name": card.details.get("set", {}).get("name", card.set_id),
            "language": card.language, "rarity": card.details.get("rarity", ""),
            "image": "/api/v1/cards/" + card.id + "/image",
            "variants": [{k: v for k, v in option.items() if k != "raw"} for option in catalog_variants(card.details)],
            "source": card.details.get("source", "TCGdex"),
            "total": card.details.get("set", {}).get("cardCount", {}).get("official"),
            "types": card.details.get("types", []),
            "reference_price": quote_payload(db, card, options[0]["id"], "NM") if db is not None and options else None}
    return result


def listing_summary(db, row, user):
    card = db.get(Card, row.card_id)
    seller = db.scalar(select(User).where(User.owner == row.owner))
    label = next((v["label"] for v in catalog_variants(card.details) if v["id"] == row.variant), row.variant)
    return {"id": row.id, "card": card_summary(card), "variant": row.variant, "variant_label": label,
            "condition": row.condition, "asking_vnd": row.asking_vnd, "status": row.status,
            "is_owner": row.owner == user, "seller": seller.display_name if seller else "Collector",
            "created_at": row.created_at, "image": f"/api/v1/listings/{row.id}/photos/front",
            "back_image": f"/api/v1/listings/{row.id}/photos/back"}


class CatalogCoverage(BaseModel):
    """Current printing, variant, and source-image counts in the local catalog."""

    source: Literal["live_database"]
    total_printings: int
    total_variants: int
    total_with_source_front_image: int
    by_source: dict[str, dict[str, int]]
    large_index_available: bool
    large_index_active: bool
    candidate_source: Literal["CardCorpus"] | None


def _cardcorpus_gallery_available() -> bool:
    root = DATA / "models" / "cardcorpus-scanner"
    try:
        manifest_path = root / "checksums.json"
        if manifest_path.stat().st_size > 4096:
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {"pipeline.json", "index/release.json", "index/embeddings.npy", "index/rows.json",
                    "index/cards.jsonl.gz", "index/dupe-groups.json", "models/matcher.safetensors",
                    "models/heads.safetensors", "mapping.json"}
        if not isinstance(manifest, dict) or set(manifest) != required:
            return False
        resolved_root = root.resolve()
        for name, digest in manifest.items():
            if re.fullmatch(r"[a-f0-9]{64}", digest) is None:
                return False
            path = root / name
            if not path.is_file() or not path.resolve().is_relative_to(resolved_root):
                return False
        mapping = root / "mapping.json"
        digest = hashlib.sha256()
        with mapping.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == manifest["mapping.json"]
    except (OSError, ValueError):
        return False


@router.get("/shop/catalog/coverage", response_model=CatalogCoverage)
def catalog_coverage():
    summary = {}
    with Session() as db:
        for details in db.scalars(select(Card.details).execution_options(yield_per=512)):
            details = details if isinstance(details, dict) else {}
            linked = details.get("cardcorpus")
            source = ("CardCorpus" if details.get("source") == "CardCorpus"
                      or isinstance(linked, dict) and isinstance(linked.get("id"), str)
                      else details.get("source", "other"))
            source = source if source in ("CardCorpus", "TCGdex") else "other"
            counts = summary.setdefault(source, {"printings": 0, "variants": 0, "front_images": 0})
            counts["printings"] += 1
            counts["variants"] += len({option["id"] for option in catalog_variants(details)})
            counts["front_images"] += picture_id(details) is not None
    available = _cardcorpus_gallery_available()
    return CatalogCoverage(source="live_database",
        total_printings=sum(counts["printings"] for counts in summary.values()),
        total_variants=sum(counts["variants"] for counts in summary.values()),
        total_with_source_front_image=sum(counts["front_images"] for counts in summary.values()),
        by_source=summary, large_index_available=available, large_index_active=False,
        candidate_source="CardCorpus" if available else None)


@router.get("/shop/catalog")
def browse(q: str = "", set_id: str | None = None, language: str | None = None,
           ids: str | None = Query(None, max_length=16100), rare: bool = False,
           sort: Literal['featured', 'name', 'price-up', 'price-down'] = 'featured',
           offset: int = Query(0, ge=0), limit: int = Query(24, ge=1, le=100)):
    with Session() as db:
        statement = select(Card)
        if q:
            statement = statement.where(Card.name.ilike("%" + q[:100] + "%") | (Card.number == q[:40]))
        if set_id:
            statement = statement.where(Card.set_id == set_id)
        if language:
            statement = statement.where(Card.language == language)
        if ids is not None:
            selected = list(dict.fromkeys(ids.split(',')))
            if len(selected) > 100:
                raise HTTPException(422, 'maximum_100_card_ids')
            statement = statement.where(Card.id.in_(selected))
        if rare:
            rarity = Card.details['rarity'].as_string()
            statement = statement.where(rarity.ilike('%rare%') | rarity.ilike('%illustration%') | rarity.ilike('%ultra%') | rarity.ilike('%hyper%'))
        total = db.scalar(select(func.count()).select_from(statement.subquery()))
        if sort in ('price-up', 'price-down'):
            # shortcut: sort canonical quotes for the local demo; materialize quotes before catalog results exceed 3000.
            if total > 3000:
                raise HTTPException(422, 'Lọc theo bộ thẻ trước khi sắp xếp giá cho hơn 3000 kết quả.')
            summaries = [card_summary(card, db) for card in db.scalars(statement.order_by(Card.name, Card.id))]
            def price_key(card):
                amount = (card.get('reference_price') or {}).get('amount_vnd')
                return (amount is None, -(amount or 0) if sort == 'price-down' else (amount or 0), card['name'], card['id'])
            summaries.sort(key=price_key)
            return {'items': summaries[offset:offset + limit], 'total': total, 'offset': offset, 'limit': limit}
        elif sort == 'name':
            ordering = (Card.name, Card.id)
        elif sort == 'featured':
            rarity = Card.details['rarity'].as_string()
            ordering = (case((rarity.ilike('%illustration%'), 0), (rarity.ilike('%special%'), 1), else_=2), Card.name, Card.id)
        rows = db.scalars(statement.order_by(*ordering).offset(offset).limit(limit)).all()
        return {"items": [card_summary(c, db) for c in rows], "total": total, "offset": offset, "limit": limit}


@router.get("/shop/sets")
def sets():
    totals = (select(Card.set_id.label("set_id"), func.min(Card.id).label("first_id"),
                     func.count(Card.id).label("count")).group_by(Card.set_id).subquery())
    with Session() as db:
        name = Card.details["set"]["name"].as_string()
        rows = db.execute(select(Card.set_id, Card.language, name, totals.c.count)
                          .join(totals, (totals.c.set_id == Card.set_id) & (totals.c.first_id == Card.id))
                          .order_by(Card.set_id)).all()
        return [{"id": set_id, "name": name or set_id, "count": count, "language": language}
                for set_id, language, name, count in rows]


@router.get("/shop/cards/{card_id}")
def detail(card_id: str, user=Depends(owner)):
    with Session() as db:
        card = db.get(Card, card_id)
        if not card:
            raise HTTPException(404, "card_not_found")
        summary = card_summary(card, db)
    if os.getenv("PKMLENS_LIVE_PRICE_FETCH") == "1" and (summary.get("reference_price") or {}).get("amount_vnd") is None:
        options = summary.get("variants") or []
        first_variant = options[0]["id"] if options else None
        if first_variant and fetch_live_price_for_card(card_id, variant=first_variant):
            with Session() as db:
                card = db.get(Card, card_id)
                summary = card_summary(card, db)
    with Session() as db:
        card = db.get(Card, card_id)
        listings = db.scalars(select(Listing).where(Listing.card_id == card.id, Listing.status == "active")
                             .order_by(Listing.asking_vnd).limit(40)).all()
        return {"card": summary, "listings": [listing_summary(db, row, user) for row in listings]}


@router.get('/shop/mine')
def mine(user=Depends(owner)):
    with Session() as db:
        rows = db.scalars(select(Listing).where(Listing.owner == user).order_by(Listing.created_at.desc())).all()
        return [listing_summary(db, row, user) for row in rows]


@router.get("/shop/listings")
def listings(q: str = "", set_id: str | None = None, condition: str | None = None,
             maximum_vnd: int | None = Query(None, ge=1), user=Depends(owner)):
    from .market import expire_reservations, lock
    with lock, Session.begin() as db:
        expire_reservations(db)
        statement = select(Listing).join(Card).where(Listing.status == "active")
        if q:
            statement = statement.where(Card.name.ilike("%" + q[:100] + "%"))
        if set_id:
            statement = statement.where(Card.set_id == set_id)
        if condition:
            statement = statement.where(Listing.condition == condition)
        if maximum_vnd is not None:
            statement = statement.where(Listing.asking_vnd <= maximum_vnd)
        return [listing_summary(db, row, user) for row in db.scalars(statement.order_by(Listing.created_at.desc()).limit(100))]


@router.get("/shop/listings/{listing_id}")
def listing_detail(listing_id: str, user=Depends(owner)):
    with Session() as db:
        row = db.get(Listing, listing_id)
        if not row or (row.status not in ("active", "reserved", "sold") and row.owner != user):
            raise HTTPException(404, "listing_not_found")
        return {**listing_summary(db, row, user),
                "reference_price": quote_payload(db, db.get(Card, row.card_id), row.variant, row.condition)}


class PriceRequest(BaseModel):
    card_id: str = Field(max_length=160)
    variant: str = Field(max_length=180)
    condition: Literal["NM", "LP", "MP", "HP", "DMG"] = "NM"


class PriceBatch(BaseModel):
    items: list[PriceRequest] = Field(min_length=1, max_length=50)


@router.post("/prices/batch")
def batch_prices(body: PriceBatch, user=Depends(owner)):
    live_fetch = os.getenv("PKMLENS_LIVE_PRICE_FETCH") == "1"
    with Session() as db:
        result = []
        for item in body.items:
            card = db.get(Card, item.card_id)
            if not card or item.variant not in {v["id"] for v in catalog_variants(card.details)}:
                raise HTTPException(422, "invalid_card_variant")
            quote = quote_payload(db, card, item.variant, item.condition)
            if live_fetch and quote.get("amount_vnd") is None:
                if fetch_live_price_for_card(card.id, variant=item.variant):
                    quote = quote_payload(db, card, item.variant, item.condition)
            result.append(quote)
        return result


class ScanRecordRequest(BaseModel):
    card_id: str = Field(max_length=160)
    variant: str | None = Field(default=None, max_length=180)
    condition: Literal["NM", "LP", "MP", "HP", "DMG"] = "NM"
    similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str = Field(default="camera", max_length=40)
    auto_added: bool = True
    fetch_price: bool = True


@router.post("/scan/record")
def record_scan_card(body: ScanRecordRequest, user=Depends(owner)):
    """Persist a recognized card into the system DB and fetch/cache its live online price."""
    with Session() as db:
        card = db.get(Card, body.card_id)
        if not card:
            raise HTTPException(404, "card_not_found")
        options = catalog_variants(card.details)
        option_ids = {v["id"]: v["label"] for v in options}
        if body.variant is not None:
            if body.variant not in option_ids:
                raise HTTPException(422, "invalid_card_variant")
            variant_id, variant_label = body.variant, option_ids[body.variant]
        elif options:
            variant_id, variant_label = options[0]["id"], options[0]["label"]
        else:
            variant_id, variant_label = "normal", "Normal"
        quote = quote_payload(db, card, variant_id, body.condition)

    price_fetched = False
    if body.fetch_price and quote.get("amount_vnd") is None and os.getenv("PKMLENS_LIVE_PRICE_FETCH") == "1":
        if fetch_live_price_for_card(body.card_id, variant=variant_id):
            price_fetched = True
            with Session() as db:
                card = db.get(Card, body.card_id)
                quote = quote_payload(db, card, variant_id, body.condition)

    added_to_collection = False
    scan_id, quote_id = secrets.token_hex(16), secrets.token_hex(16)
    with Session.begin() as db:
        card = db.get(Card, body.card_id)
        db.add(Scan(
            id=scan_id, owner=user, created_at=now(),
            result={"mode": "live_scanner", "source": body.source, "auto_added": body.auto_added,
                    "similarity": body.similarity},
            card_id=card.id, variant=variant_id, condition=body.condition, revision=1,
        ))
        db.flush()
        db.add(Quote(
            id=quote_id, scan_id=scan_id, owner=user, revision=1, created_at=now(), payload=quote,
        ))
        is_member = db.scalar(select(User.id).where(User.owner == user)) is not None
        if is_member:
            existing = db.scalar(
                select(PersonalCard).where(
                    PersonalCard.owner == user,
                    PersonalCard.card_id == card.id,
                    PersonalCard.variant == variant_id,
                    PersonalCard.kind == "collection",
                )
            )
            if existing is None:
                db.add(PersonalCard(
                    id=secrets.token_hex(16), owner=user, card_id=card.id,
                    variant=variant_id, kind="collection", quantity=1,
                ))
            else:
                existing.quantity = min(10000, existing.quantity + 1)
            added_to_collection = True
    return {
        "scan_id": scan_id,
        "card_id": body.card_id,
        "variant": variant_id,
        "variant_label": variant_label,
        "added_to_collection": added_to_collection,
        "price_fetched": price_fetched,
        "quote": quote,
    }
