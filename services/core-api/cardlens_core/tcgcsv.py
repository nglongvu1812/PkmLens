"""TCGCSV snapshots joined only through reviewed TCGplayer product mappings."""
import json
import os
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation
import httpx
from sqlalchemy import select
from .catalog import now,key,variants
from .db import DATA,Session,Card,Mapping,Price

SUBTYPES={"normal":"normal","holo":"holofoil","reverse":"reverseholofoil"}

def ingest_prices(group_id,rows,source_at,fetched_at):
    timestamp=datetime.fromisoformat(source_at).astimezone(timezone.utc).isoformat()
    saved=0
    with Session.begin() as db:
        for row in rows:
            mappings=db.scalars(select(Mapping).where(Mapping.provider=="tcgplayer",Mapping.provider_id==str(row["productId"]),Mapping.status=="matched")).all()
            if len(mappings)!=1:continue
            mapping=mappings[0]
            card=db.get(Card,mapping.card_id)
            variant=next((v for v in variants(card.details) if v["id"]==mapping.variant),None)
            if not variant:continue
            expected=SUBTYPES.get(variant["raw"].get("type",variant["id"]))
            subtype=row.get("subTypeName","").lower().replace(" ","").replace("-","")
            if expected!=subtype:continue
            value=row.get("marketPrice")
            if value is None:value=row.get("midPrice")
            try:
                amount=Decimal(str(value))
                if not amount.is_finite() or amount<=0:continue
            except InvalidOperation:continue
            source="TCGCSV/tcgplayer"
            ident=key(mapping.card_id+mapping.variant+source+timestamp)
            db.merge(Price(id=ident,card_id=mapping.card_id,variant=mapping.variant,source=source,source_at=timestamp,fetched_at=fetched_at,currency="USD",amount=str(amount),low=str(row["lowPrice"]) if row.get("lowPrice") is not None else None,high=str(row["highPrice"]) if row.get("highPrice") is not None else None,evidence={"group_id":group_id,"product_id":row["productId"],"variant_id":mapping.variant,"subtype":row["subTypeName"],"upstream":"tcgplayer","mapping_status":"matched","raw":row}))
            saved+=1
    return saved

def sync_tcgcsv(group_ids=None):
    base=os.getenv("TCGCSV_URL","https://tcgcsv.com").rstrip("/")
    root=DATA/"catalog"/"tcgcsv"
    root.mkdir(parents=True,exist_ok=True)
    with httpx.Client(timeout=45,follow_redirects=True) as client:
        if group_ids is None:
            response=client.get(base+"/tcgplayer/3/groups");response.raise_for_status()
            with Session() as db:
                names={c.details["set"]["name"].casefold() for c in db.scalars(select(Card))}
            group_ids=[g["groupId"] for g in response.json()["results"] if g["name"].casefold() in names]
        stamp=client.get(base+"/last-updated.txt");stamp.raise_for_status()
        source_at=datetime.fromisoformat(stamp.text.strip()).isoformat()
        counts={}
        for group in group_ids:
            response=client.get(f"{base}/tcgplayer/3/{int(group)}/prices");response.raise_for_status()
            data=response.json()
            if not data.get("success"):raise ValueError("tcgcsv_upstream_failed")
            fetched=now()
            (root/(str(group)+"-"+source_at[:10]+".json")).write_text(json.dumps({"source":str(response.url),"source_at":source_at,"fetched_at":fetched,"payload":data}),encoding="utf-8")
            counts[str(group)]=ingest_prices(group,data["results"],source_at,fetched)
        return counts

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--groups",nargs="*",type=int)
    print(sync_tcgcsv(p.parse_args().groups))
