from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime,timezone
from sqlalchemy import select
from .db import Price, Fx, Mapping

CONDITIONS = {"NM": "1", "LP": "0.85", "MP": "0.70", "HP": "0.50", "DMG": "0.35"}

def calculate(amount, rate, condition):
    return int((Decimal(amount) * Decimal(rate) * Decimal(CONDITIONS[condition])).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def quote_payload(db, card, variant, condition):
    candidates=db.scalars(select(Price).where(Price.card_id==card.id,Price.variant==variant).order_by(Price.source_at.desc())).all()
    prices=[]
    for price in candidates:
        provider=price.source.split("/")[-1]
        mapping=db.scalar(select(Mapping).where(Mapping.provider==provider,Mapping.card_id==price.card_id,Mapping.variant==price.variant,Mapping.status=="matched"))
        if mapping and price.evidence.get("mapping_status","matched")=="matched":
            prices.append(price)
    fx=db.scalar(select(Fx).order_by(Fx.source_at.desc()))
    payload={"card_id":card.id,"variant":variant,"condition":condition,"amount_vnd":None,"reason":"missing_variant_price","condition_multiplier":CONDITIONS[condition],"condition_policy":"experimental_not_market_validated","rounding":"VND_integer_HALF_UP","disclaimer":"Reference price, not guaranteed","source":None,"source_at":None,"fx_source":fx.source if fx else None,"fx_at":fx.source_at if fx else None}
    if not prices:return payload
    price=next((p for p in prices if p.source=="TCGdex/tcgplayer"),next((p for p in prices if p.source=="TCGCSV/tcgplayer"),prices[0]))
    payload.update(source=price.source,source_at=price.source_at,fetched_at=price.fetched_at,original_amount=price.amount,currency=price.currency,price_snapshot_id=price.id,upstream=price.source.split("/")[-1],independent_source_count=1)
    if not fx:
        payload["reason"]="missing_fx"
        return payload
    try:
        vnd=Decimal(fx.rates["VND"])
        base=Decimal(fx.rates[price.currency])
        if not vnd.is_finite() or not base.is_finite() or vnd<=0 or base<=0:raise ValueError("invalid_fx")
        rate=vnd/base
        price_date=datetime.fromisoformat(price.source_at.replace("Z","+00:00"))
        if price_date.tzinfo is None:price_date=price_date.replace(tzinfo=timezone.utc)
        payload["price_stale"]=(datetime.now(timezone.utc)-price_date).total_seconds()>72*3600
        payload["fx_stale"]=(datetime.now(timezone.utc).timestamp()-int(fx.source_at))>48*3600
    except (KeyError,InvalidOperation,ValueError,ZeroDivisionError):
        payload["reason"]="invalid_price_or_fx_snapshot"
        return payload
    payload.update(amount_vnd=calculate(price.amount,str(rate),condition),fx_rate=str(rate),reason=None,range_vnd=[calculate(price.low or price.amount,str(rate),condition),calculate(price.high or price.amount,str(rate),condition)])
    return payload
