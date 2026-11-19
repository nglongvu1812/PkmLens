"""On-demand live online price lookup (TCGdex -> TCGplayer USD / Cardmarket EUR) with DB caching."""
import difflib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import httpx
from sqlalchemy import select
from .catalog import catalog_variants, key, now, sync_fx
from .db import Card, Fx, Mapping, Price, Session

log = logging.getLogger(__name__)
_fetch_lock = threading.Lock()
_recent_attempts: dict[str, float] = {}
_COOLDOWN_SECONDS = 600.0


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).strip()


def _pick_tcgplayer_bucket(tcgplayer: dict, variant_id: str) -> tuple[str, dict] | None:
    buckets = {k: v for k, v in tcgplayer.items() if isinstance(v, dict)}
    if not buckets:
        return None
    v = _norm(variant_id)
    preferences = []
    if "reverse" in v:
        preferences.extend(["reverse-holofoil", "reverse", "holofoil", "normal"])
    elif "1st" in v or "first" in v:
        preferences.extend(["1st-edition-holofoil", "1st-edition-normal", "holofoil", "normal"])
    elif "holo" in v or "foil" in v:
        preferences.extend(["holofoil", "normal", "reverse-holofoil"])
    else:
        preferences.extend(["normal", "holofoil", "reverse-holofoil"])
    for name in preferences:
        if name in buckets:
            bucket = buckets[name]
            if (bucket.get("marketPrice") or bucket.get("midPrice") or bucket.get("lowPrice")) is not None:
                return name, bucket
    for name, bucket in buckets.items():
        if (bucket.get("marketPrice") or bucket.get("midPrice") or bucket.get("lowPrice")) is not None:
            return name, bucket
    return None


def _ensure_fresh_fx() -> None:
    try:
        with Session() as db:
            fx = db.scalar(select(Fx).order_by(Fx.source_at.desc()))
        stale = True
        if fx is not None:
            try:
                stale = (datetime.now(timezone.utc).timestamp() - int(fx.source_at)) > 48 * 3600
            except (ValueError, TypeError):
                stale = True
        if stale:
            sync_fx()
    except Exception:
        log.debug("live_price_fx_refresh_skipped", exc_info=True)


def _search_tcgdex_detail(client: httpx.Client, card: Card) -> tuple[str, dict] | None:
    base = os.getenv("TCGDEX_BASE_URL", "https://api.tcgdex.net/v2").rstrip("/")
    langs = []
    raw_lang = (card.language or "en").lower()
    if raw_lang in ("en", "ja", "fr", "de", "es", "it", "pt"):
        langs.append(raw_lang)
    if "en" not in langs:
        langs.append("en")

    target_set = _norm((card.details or {}).get("set", {}).get("name") or card.set_id)
    target_name = _norm(card.name)
    num_candidates = []
    if card.number:
        num_candidates.append(str(card.number).strip())
        stripped = str(card.number).strip().lstrip("0")
        if stripped and stripped not in num_candidates:
            num_candidates.append(stripped)
        if not stripped and "0" not in num_candidates:
            num_candidates.append("0")

    for lang in langs:
        hits = []
        for num in num_candidates:
            try:
                resp = client.get(f"{base}/{lang}/cards", params={"name": card.name, "localId": num})
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        hits = data
                        break
            except Exception:
                continue
        if not hits:
            try:
                resp = client.get(f"{base}/{lang}/cards", params={"name": card.name})
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        if num_candidates:
                            exact_num = [h for h in data if str(h.get("localId", "")).lstrip("0") in {n.lstrip("0") for n in num_candidates}]
                            hits = exact_num or data[:5]
                        else:
                            hits = data[:5]
            except Exception:
                continue
        if not hits:
            continue

        best_detail = None
        best_score = -1.0
        for brief in hits[:6]:
            cid = brief.get("id")
            if not cid:
                continue
            try:
                d_resp = client.get(f"{base}/{lang}/cards/{cid}")
                if d_resp.status_code != 200:
                    continue
                detail = d_resp.json()
            except Exception:
                continue
            pricing = detail.get("pricing") or {}
            if not pricing.get("tcgplayer") and not pricing.get("cardmarket"):
                continue
            set_name = _norm((detail.get("set") or {}).get("name") or "")
            name_sim = difflib.SequenceMatcher(None, target_name, _norm(detail.get("name"))).ratio()
            set_sim = difflib.SequenceMatcher(None, target_set, set_name).ratio() if target_set and set_name else 0.0
            if target_set and set_name and (set_name in target_set or target_set in set_name):
                set_sim = max(set_sim, 0.92)
            num_bonus = 0.35 if num_candidates and str(detail.get("localId", "")).lstrip("0") in {n.lstrip("0") for n in num_candidates} else 0.0
            score = name_sim * 0.4 + set_sim * 0.6 + num_bonus
            if score > best_score:
                best_score = score
                best_detail = (f"{base}/{lang}/cards/{cid}", detail)
        if best_detail is not None:
            return best_detail
    return None


def fetch_live_price_for_card(card_id: str, variant: str | None = None, *, force: bool = False) -> bool:
    """Fetch live TCGplayer/Cardmarket pricing via TCGdex and persist into SQLite price_snapshots."""
    with Session() as db:
        card = db.get(Card, card_id)
        if not card:
            return False
        options = catalog_variants(card.details)
        target_variants = [variant] if variant else ([options[0]["id"]] if options else ["normal"])
        if not force:
            existing = db.scalar(
                select(Price).where(Price.card_id == card.id, Price.variant == target_variants[0]).order_by(Price.source_at.desc())
            )
            if existing is not None:
                try:
                    dt = datetime.fromisoformat(existing.fetched_at.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if (datetime.now(timezone.utc) - dt).total_seconds() < 12 * 3600:
                        return True
                except ValueError:
                    pass

    cache_key = f"{card_id}:{target_variants[0]}"
    now_mono = time.monotonic()
    with _fetch_lock:
        if not force and now_mono - _recent_attempts.get(cache_key, 0.0) < _COOLDOWN_SECONDS:
            return False
        _recent_attempts[cache_key] = now_mono

    _ensure_fresh_fx()

    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            found = _search_tcgdex_detail(client, card)
    except Exception:
        log.warning("live_price_lookup_failed card=%s", card_id, exc_info=True)
        return False

    if not found:
        return False

    url, detail = found
    pricing = detail.get("pricing") or {}
    fetched_at = now()
    saved = False

    with Session.begin() as db:
        for variant_id in target_variants:
            tcg = pricing.get("tcgplayer")
            if isinstance(tcg, dict) and tcg.get("unit") in ("USD", "EUR"):
                picked = _pick_tcgplayer_bucket(tcg, variant_id)
                if picked:
                    bucket_name, bucket = picked
                    raw_amount = bucket.get("marketPrice") or bucket.get("midPrice") or bucket.get("lowPrice")
                    try:
                        amount = Decimal(str(raw_amount))
                        if amount.is_finite() and amount > 0:
                            product_id = str(bucket.get("productId") or detail.get("id") or card.id)
                            source_at = str(tcg.get("updated") or fetched_at)
                            existing_map = db.scalar(
                                select(Mapping).where(
                                    Mapping.provider == "tcgplayer",
                                    Mapping.provider_id == product_id,
                                    Mapping.variant == variant_id,
                                    Mapping.card_id == card.id,
                                )
                            )
                            map_id = existing_map.id if existing_map else key(json.dumps(["tcgplayer", product_id, variant_id, card.id]))
                            db.merge(Mapping(id=map_id, card_id=card.id, provider="tcgplayer", provider_id=product_id, variant=variant_id, status="matched"))
                            db.flush()
                            price_id = key(card.id + variant_id + "TCGdex/tcgplayer" + source_at)
                            low_val = bucket.get("lowPrice")
                            high_val = bucket.get("highPrice")
                            db.merge(
                                Price(
                                    id=price_id,
                                    card_id=card.id,
                                    variant=variant_id,
                                    source="TCGdex/tcgplayer",
                                    source_at=source_at,
                                    fetched_at=fetched_at,
                                    currency=str(tcg["unit"]),
                                    amount=str(amount),
                                    low=str(low_val) if low_val is not None else None,
                                    high=str(high_val) if high_val is not None else None,
                                    evidence={
                                        "url": url,
                                        "tcgdex_id": detail.get("id"),
                                        "bucket": bucket_name,
                                        "product_id": product_id,
                                        "mapping_status": "matched",
                                        "raw": bucket,
                                    },
                                )
                            )
                            saved = True
                            continue
                    except InvalidOperation:
                        pass

            cm = pricing.get("cardmarket")
            if isinstance(cm, dict) and cm.get("unit") in ("USD", "EUR"):
                v_norm = _norm(variant_id)
                raw_amount = (
                    (cm.get("trend-holo") or cm.get("avg-holo")) if "holo" in v_norm else None
                ) or cm.get("trend") or cm.get("avg") or cm.get("avg30") or cm.get("low")
                if raw_amount is not None:
                    try:
                        amount = Decimal(str(raw_amount))
                        if amount.is_finite() and amount > 0:
                            product_id = str(cm.get("idProduct") or detail.get("id") or card.id)
                            source_at = str(cm.get("updated") or fetched_at)
                            existing_map = db.scalar(
                                select(Mapping).where(
                                    Mapping.provider == "cardmarket",
                                    Mapping.provider_id == product_id,
                                    Mapping.variant == variant_id,
                                    Mapping.card_id == card.id,
                                )
                            )
                            map_id = existing_map.id if existing_map else key(json.dumps(["cardmarket", product_id, variant_id, card.id]))
                            db.merge(Mapping(id=map_id, card_id=card.id, provider="cardmarket", provider_id=product_id, variant=variant_id, status="matched"))
                            db.flush()
                            price_id = key(card.id + variant_id + "TCGdex/cardmarket" + source_at)
                            low_val = cm.get("low")
                            db.merge(
                                Price(
                                    id=price_id,
                                    card_id=card.id,
                                    variant=variant_id,
                                    source="TCGdex/cardmarket",
                                    source_at=source_at,
                                    fetched_at=fetched_at,
                                    currency=str(cm["unit"]),
                                    amount=str(amount),
                                    low=str(low_val) if low_val is not None else None,
                                    high=None,
                                    evidence={
                                        "url": url,
                                        "tcgdex_id": detail.get("id"),
                                        "product_id": product_id,
                                        "mapping_status": "matched",
                                        "raw": cm,
                                    },
                                )
                            )
                            saved = True
                    except InvalidOperation:
                        pass

    return saved
