import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from decimal import Decimal
import httpx
from sqlalchemy import func, select
from .db import DATA, Session, Card, Mapping, Price, Fx

def now():
    return datetime.now(timezone.utc).isoformat()

def key(value):
    return hashlib.sha256(value.encode()).hexdigest()

def variants(detail):
    detailed = detail.get("variants_detailed") or []
    if detailed:
        return [{"id": v["variantId"], "label": " / ".join([v.get("type", "unknown"), v.get("subtype", ""), ",".join(v.get("stamp", []))]).strip(" /"), "raw": v} for v in detailed if v.get("variantId")]
    return [{"id": k, "label": k, "raw": {}} for k in ("normal", "holo", "reverse") if detail.get("variants", {}).get(k)]

def catalog_variants(details):
    """Local variants plus source-only options, which remain unverified."""
    details = details if isinstance(details, dict) else {}
    options = variants(details)
    metadata = details.get("cardcorpus")
    provenance = metadata.get("id") if isinstance(metadata, dict) else None
    source_options = (metadata.get("source_variant_options", []) if isinstance(provenance, str)
                      and re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", provenance)
                      else [])
    source = details.get("source", "TCGdex")
    if not isinstance(source, str):
        source = "TCGdex"
    known = set()
    for option in options:
        if option["id"] in known:
            continue
        known.add(option["id"])
        option["variant_id"] = option["id"]
        option["variant_source"] = source
        if source == "CardCorpus":
            option["verified"] = False
    if isinstance(source_options, list):
        for source_option in source_options:
            ident = source_option.get("id", source_option.get("variantId")) if isinstance(source_option, dict) else source_option
            if (not isinstance(ident, str) or ident.strip() != ident or not 0 < len(ident) <= 180
                    or not all(char.isprintable() for char in ident) or ident in known):
                continue
            variant_type = source_option.get("type", "source_only") if isinstance(source_option, dict) else "source_only"
            if not isinstance(variant_type, str) or not variant_type.isprintable() or not 0 < len(variant_type) <= 64:
                variant_type = "source_only"
            options.append({"id": ident, "label": ident, "type": variant_type, "source": "CardCorpus",
                            "variant_id": ident, "variant_source": "CardCorpus", "verified": False})
            known.add(ident)
    return options


def save_card(detail, fetched):
    cid = "{}:{}:en".format(detail["set"]["id"], detail["localId"])
    detailed = variants(detail)
    provider_usage = {}
    for item in detailed:
        for provider, provider_id in (item["raw"].get("thirdParty") or {}).items():
            provider_usage.setdefault((provider, str(provider_id)), set()).add(item["id"])
    ambiguous = {key for key, value in provider_usage.items() if len(value) > 1}
    with Session.begin() as db:
        existing = db.get(Card, cid)
        merged = dict(detail)
        if existing:
            if existing.provider_id != detail["id"] or existing.language != "en":
                raise ValueError("catalog_provider_identity_conflict")
            for field in ("cardcorpus", "cardcorpus_id", "number_is_printed", "picture_id", "image_url"):
                if field in existing.details:
                    merged[field] = existing.details[field]
        db.merge(Card(id=cid, provider_id=detail["id"], set_id=detail["set"]["id"], number=detail["localId"], language="en", name=detail["name"], image=detail.get("image", ""), details=merged))
        db.flush()
        db.merge(Mapping(id=key("tcgdex:" + detail["id"]), card_id=cid, provider="tcgdex", provider_id=detail["id"], variant="", status="matched"))
        for variant in detailed:
            raw = variant["raw"]
            # shortcut: legacy undifferentiated prices are not safe variant evidence; leave them unavailable.
            pricing = raw.get("pricing") or {}
            for provider, provider_id in (raw.get("thirdParty") or {}).items():
                existing=db.scalar(select(Mapping).where(Mapping.provider==provider,Mapping.provider_id==str(provider_id),Mapping.variant==variant["id"],Mapping.card_id==cid))
                ident=existing.id if existing else key(json.dumps([provider,str(provider_id),variant["id"],cid]))
                db.merge(Mapping(id=ident,card_id=cid,provider=provider,provider_id=str(provider_id),variant=variant["id"],status="unmatched" if (provider,str(provider_id)) in ambiguous else "matched"))
            for source, prices in pricing.items():
                provider_id = str((raw.get("thirdParty") or {}).get(source, ""))
                if not prices or (source, provider_id) in ambiguous:
                    continue
                unit = prices.get("unit")
                entries = [p for p in prices.values() if isinstance(p, dict)] if source == "tcgplayer" else [prices]
                for entry in entries:
                    amount = entry.get("marketPrice") or entry.get("midPrice") if source == "tcgplayer" else entry.get("trend")
                    if amount is None or unit not in ("USD", "EUR"):
                        continue
                    source_at = prices.get("updated", fetched)
                    ident = key(cid + variant["id"] + source + source_at)
                    db.merge(Price(id=ident, card_id=cid, variant=variant["id"], source="TCGdex/" + source, source_at=source_at, fetched_at=fetched, currency=unit, amount=str(amount), low=str(entry.get("lowPrice", entry.get("low"))) if entry.get("lowPrice", entry.get("low")) is not None else None, high=str(entry["highPrice"]) if entry.get("highPrice") is not None else None, evidence={"provider_id": detail["id"], "variant_id": variant["id"], "pricing": prices, "mapping_status": "unmatched" if (source, provider_id) in ambiguous else "matched"}))
    return cid

def sync_set(set_id="base1", limit=None, refresh=False, *, provider_ids=None):
    if not set_id.replace("-", "").isalnum():
        raise ValueError("Invalid set id")
    root = DATA / "catalog"
    root.mkdir(exist_ok=True)
    gallery = DATA / "gallery"
    gallery.mkdir(exist_ok=True)
    base = os.getenv("TCGDEX_URL", "https://api.tcgdex.net/v2/en")
    manifest = []
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        response = client.get(base + "/sets/" + set_id)
        response.raise_for_status()
        cards = response.json()["cards"]
        if provider_ids is not None:
            cards = [brief for brief in cards if brief["id"] in provider_ids]
        for brief in cards[:limit]:
            path = root / (brief["id"] + ".json")
            if refresh or not path.exists():
                r = client.get(base + "/cards/" + brief["id"])
                r.raise_for_status()
                detail = r.json()
                path.write_text(json.dumps({"fetched_at": now(), "detail": detail}, ensure_ascii=False), encoding="utf-8")
                time.sleep(0.12)
            cached = json.loads(path.read_text(encoding="utf-8"))
            cid = save_card(cached["detail"], cached["fetched_at"])
            image = gallery / (brief["id"] + ".webp")
            url = (cached["detail"].get("image") or "") + "/high.webp"
            if not image.exists() and cached["detail"].get("image"):
                r = client.get(url)
                r.raise_for_status()
                image.write_bytes(r.content)
            if image.exists():
                manifest.append({"card_id": cid, "provider_id": brief["id"], "path": str(image), "source": url, "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "fetched_at": cached["fetched_at"]})
            print("synced", cid, flush=True)
    reconcile_provider_mappings()
    (root / (set_id + "-manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(manifest)

def sync_fx():
    url = os.getenv("FX_URL", "https://open.er-api.com/v6/latest/USD")
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    value = response.json()
    rates = value["rates"]
    if value.get("result") != "success" or Decimal(str(rates.get("VND", 0))) <= 0:
        raise ValueError("Invalid FX response")
    source_at = str(value["time_last_update_unix"])
    with Session.begin() as db:
        db.merge(Fx(id=key(url + source_at), source_at=source_at, fetched_at=now(), source=url, rates={k: str(v) for k, v in rates.items()}))
    return source_at

def refresh_prices():
    """Refresh only known TCGdex printings, not the imported CardCorpus sets."""
    sets = {}
    with Session() as db:
        for set_id, provider_id in db.execute(select(Card.set_id, Mapping.provider_id)
                .join(Mapping, Mapping.card_id == Card.id).where(
                    Mapping.provider == "tcgdex", Mapping.status == "matched", Mapping.variant == "",
                    Card.provider_id == Mapping.provider_id,
                    func.coalesce(Card.details["source"].as_string(), "TCGdex") == "TCGdex")):
            sets.setdefault(set_id, set()).add(provider_id)
    total = sum(sync_set(set_id, refresh=True, provider_ids=sets[set_id]) for set_id in sorted(sets))
    fx_at=sync_fx()
    from .tcgcsv import sync_tcgcsv
    try:
        secondary={"status":"available","groups":sync_tcgcsv()}
    except (httpx.HTTPError,ValueError) as exc:
        status=exc.response.status_code if isinstance(exc,httpx.HTTPStatusError) else None
        secondary={"status":"unavailable","http_status":status,"error":"upstream_request_failed" if isinstance(exc,httpx.HTTPError) else "invalid_upstream_response"}
    return {"cards":total,"fx":fx_at,"tcgcsv":secondary,"alerts_triggered":evaluate_alerts()}


def evaluate_alerts():
    """Alerts are one-shot: crossing creates exactly one durable notification."""
    from .db import PriceAlert,Notification
    from .pricing import quote_payload
    count=0
    with Session.begin() as db:
        alerts=db.scalars(select(PriceAlert).where(PriceAlert.active.is_(True)).with_for_update()).all()
        for alert in alerts:
            card=db.get(Card,alert.card_id)
            if not card:
                continue
            quote=quote_payload(db,card,alert.variant,"NM")
            amount=quote.get("amount_vnd")
            if amount is None or quote.get("price_stale") or quote.get("fx_stale"):
                continue
            crossed=amount<=alert.threshold_vnd if alert.direction=="below" else amount>=alert.threshold_vnd
            if crossed:
                ident=key(alert.owner+alert.id)
                if db.get(Notification,ident) is None:
                    db.add(Notification(id=ident,owner=alert.owner,alert_id=alert.id,created_at=now(),payload={"card_id":alert.card_id,"variant":alert.variant,"amount_vnd":amount,"threshold_vnd":alert.threshold_vnd,"fx_at":quote.get("fx_at"),"source_at":quote.get("source_at")}))
                    count+=1
                alert.active=False
                alert.triggered_at=now()
    return count


def reconcile_provider_mappings():
    """Do not quote provider IDs reused across distinct cards or variants."""
    from .db import Mapping
    with Session.begin() as db:
        mappings=db.scalars(select(Mapping).where(Mapping.provider!="tcgdex",Mapping.variant!="")).all()
        groups={}
        for mapping in mappings:
            groups.setdefault((mapping.provider,mapping.provider_id),set()).add((mapping.card_id,mapping.variant))
        ambiguous={key for key,pairs in groups.items() if len(pairs)>1}
        for mapping in mappings:
            mapping.status="unmatched" if (mapping.provider,mapping.provider_id) in ambiguous else "matched"
        prices=db.scalars(select(Price)).all()
        for price in prices:
            provider=price.source.split("/")[-1]
            mapping=db.scalar(select(Mapping).where(Mapping.provider==provider,Mapping.card_id==price.card_id,Mapping.variant==price.variant,Mapping.status=="matched"))
            price.evidence={**price.evidence,"mapping_status":"matched" if mapping else "unmatched"}
    return len(ambiguous)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="base1")
    p.add_argument("--limit", type=int)
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    print("Imported", sync_set(a.set, a.limit, a.refresh))
    print("FX", sync_fx())
