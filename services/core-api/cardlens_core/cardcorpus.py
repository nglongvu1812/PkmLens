"""Idempotently import sourced CardCorpus printing metadata into local records."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re

from sqlalchemy import select

from .catalog import key, now
from .db import Card, Mapping, Session

log = logging.getLogger(__name__)
SET_ALIASES = {"international/base-set": "base1", "international/base-set-2": "base4"}
CATALOGS = {"international", "japanese", "zh-hans"}
PICTURE_TEMPLATE = "https://img.cardcorpus.com/s1/{width}/{id}.avif"
PICTURE_ID = re.compile(r"[0-9a-f]{32}\Z")
SET_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
CARD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
LANGUAGE = re.compile(r"[A-Za-z0-9-]+\Z")
SOURCE_FIELDS = (
    "id", "name", "number", "rarity", "cardType", "language", "credit", "artists", "released",
    "defaultCopy", "defaultCollectingVariation", "defaultCollectingCopy", "printRuns", "inSet",
    "eReaderLayout", "displayName",
)
VARIANT_FIELDS = (
    "id", "name", "role", "finish", "stamp", "edition", "printRun", "editionMark", "serialCode",
    "serialCodeVariantOf", "differsFromStandard", "distributions", "material", "eReaderStrip",
    "errorState", "collectingVariation",
)
MARKETPLACES = ("tcgplayer", "cardmarket")


def identity(set_id, number, language):
    return f"{SET_ALIASES.get(set_id, 'cc-' + set_id.replace('/', '-'))}:{number}:{language}"


def _contained(path, root, *, strict=True, code="invalid_cardcorpus_path"):
    try:
        path = Path(path).resolve(strict=strict)
        root = Path(root).resolve(strict=True)
    except OSError as exc:
        raise ValueError(code) from exc
    if not path.is_relative_to(root):
        raise ValueError(code)
    return path


def _picture_url(index, image_id):
    if index.get("pictures", {}).get("sized") != PICTURE_TEMPLATE:
        raise ValueError("invalid_cardcorpus_picture_template")
    if image_id is None:
        return ""
    if not isinstance(image_id, str) or not PICTURE_ID.fullmatch(image_id):
        raise ValueError("invalid_cardcorpus_picture_id")
    return PICTURE_TEMPLATE.format(width=480, id=image_id)


def _compact_picture(picture):
    if not isinstance(picture, dict):
        return None
    result = {}
    for side in ("front", "back"):
        side_data = picture.get(side)
        if not isinstance(side_data, dict) or side_data.get("image") is None:
            continue
        image_id = side_data["image"]
        if not isinstance(image_id, str) or not PICTURE_ID.fullmatch(image_id):
            raise ValueError("invalid_cardcorpus_picture_id")
        result[side] = {"image": image_id}
    return result or None


def _sources(note):
    if not isinstance(note, dict) or not isinstance(note.get("sources"), list):
        return None
    refs = [{key: item[key] for key in ("title", "url") if isinstance(item.get(key), str)}
            for item in note["sources"] if isinstance(item, dict)
            and isinstance(item.get("title"), str) and isinstance(item.get("url"), str)
            and len(item["title"]) <= 500 and len(item["url"]) <= 2000]
    return {"sources": refs} if refs else None


def _original(card, variants):
    original = {field: card[field] for field in SOURCE_FIELDS if field in card and card[field] is not None}
    picture = _compact_picture(card.get("picture"))
    if picture:
        original["picture"] = picture
    if variants:
        original["variants"] = variants
    links = card.get("links")
    if isinstance(links, dict) and isinstance(links.get("painting"), dict):
        painting = links["painting"]
        if all(isinstance(painting.get(field), str) for field in ("set", "painting")):
            original["links"] = {"painting": {field: painting[field] for field in ("$kind", "set", "painting") if field in painting}}
    for field in ("tell", "description"):
        attribution = _sources(card.get(field))
        if attribution:
            original[field] = attribution
    if isinstance(original.get("released"), dict):
        original["released"] = {field: value for field, value in original["released"].items()
                                 if field in ("date", "sources")}
    return original


def _compact_variant(variant):
    compact = {field: variant[field] for field in VARIANT_FIELDS if field in variant and variant[field] is not None}
    picture = _compact_picture(variant.get("picture"))
    if picture:
        compact["picture"] = picture
    for field in ("tell", "description"):
        attribution = _sources(variant.get(field))
        if attribution:
            compact[field] = attribution
    return compact


def _market_options(market, card_id, variant_id):
    card = market.get(card_id)
    variants = card.get("variants") if isinstance(card, dict) else None
    source = variants.get(variant_id) if isinstance(variants, dict) else None
    if not isinstance(source, dict):
        return {}
    result = {}
    for provider in MARKETPLACES:
        record = source.get(provider)
        if not isinstance(record, dict):
            continue
        option = {}
        product = record.get("product")
        if type(product) is int and product > 0:
            option["product"] = product
        subtype = record.get("subtype")
        if isinstance(subtype, str) and len(subtype) <= 80:
            option["subtype"] = subtype
        reverse_holo = record.get("reverseHolo")
        if type(reverse_holo) is bool:
            option["reverseHolo"] = reverse_holo
        if option:
            result[provider] = option
    return result


def _read_market(path, market_root, expected_set):
    if not market_root.exists():
        return {}
    path = _contained(path, market_root, strict=False, code="invalid_cardcorpus_marketplace_path")
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError("invalid_cardcorpus_marketplace_file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("set") != expected_set
            or not isinstance(data.get("cards"), dict)):
        raise ValueError("invalid_cardcorpus_marketplace_record")
    return data["cards"]


def _same_name(left, right):
    if left == right:
        return True
    # Legacy TCGdex drops this one space; both sources describe the same printed name.
    normalize = lambda value: re.sub(r"^Nidoran\s+([\u2640\u2642])$", r"Nidoran\1", value)
    return normalize(left) == normalize(right)


def _merge_missing(incoming, existing):
    """Add newly sourced keys while keeping every existing value authoritative."""
    if not isinstance(existing, dict):
        return existing if existing not in (None, "", [], {}) else incoming
    if not isinstance(incoming, dict):
        return existing
    return {field: _merge_missing(incoming.get(field), value) if field in incoming else value
            for field, value in {**incoming, **existing}.items()}


def import_snapshot(root, catalogs=("international", "japanese", "zh-hans")):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("invalid_cardcorpus_snapshot_root")
    index_path = _contained(root / "index.json", root, code="invalid_cardcorpus_snapshot_root")
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    if not isinstance(index, dict) or index.get("name") != "CardCorpus":
        raise ValueError("not_a_cardcorpus_snapshot")
    available = index.get("catalogs")
    if not isinstance(available, list) or any(item not in CATALOGS for item in available):
        raise ValueError("invalid_cardcorpus_catalog_list")
    if index.get("pictures", {}).get("sized") != PICTURE_TEMPLATE:
        raise ValueError("invalid_cardcorpus_picture_template")
    selected = tuple(dict.fromkeys(catalogs))
    if not selected or any(catalog not in available or catalog not in CATALOGS for catalog in selected):
        raise ValueError("unknown_catalog")

    selected = tuple(sorted(selected))
    snapshot_hash = hashlib.sha256(index_bytes)
    totals = {"card_records": 0, "distinct_printings": 0, "variant_options": 0,
              "front_image_records": 0, "missing_front_image_records": 0}
    by_catalog = {}
    saved = linked = 0
    set_root = _contained(root / "sets", root, code="invalid_cardcorpus_sets_path")
    market_root = _contained(root / "marketplaces", root, strict=False,
                             code="invalid_cardcorpus_marketplaces_path")

    for catalog in selected:
        catalog_root = _contained(set_root / catalog, set_root, code="invalid_cardcorpus_catalog_path")
        if not catalog_root.is_dir():
            raise ValueError("invalid_cardcorpus_catalog_path")
        summary = {key: 0 for key in totals}
        seen_sets = set()
        for candidate in sorted(catalog_root.glob("*.json")):
            path = _contained(candidate, catalog_root, code="invalid_cardcorpus_set_path")
            raw = path.read_bytes()
            data = json.loads(raw)
            external_set = f"{catalog}/{path.stem}"
            if (not isinstance(data, dict) or data.get("id") != external_set
                    or not SET_ID.fullmatch(path.stem) or not isinstance(data.get("name"), str)
                    or len(data["name"]) > 200 or not isinstance(data.get("cards"), list)):
                raise ValueError("invalid_cardcorpus_set_record")
            if external_set in seen_sets:
                raise ValueError("duplicate_cardcorpus_set_id")
            seen_sets.add(external_set)
            source_hash = hashlib.sha256(raw).hexdigest()
            relative = path.relative_to(root).as_posix()
            snapshot_hash.update(relative.encode("utf-8") + b"\0" + bytes.fromhex(source_hash))
            market_path = market_root / catalog / (path.stem + ".json")
            market = _read_market(market_path, market_root, external_set)
            prepared = []
            seen_cards = set()
            for card in data["cards"]:
                if (not isinstance(card, dict) or not isinstance(card.get("id"), str)
                        or not CARD_ID.fullmatch(card["id"]) or len(card["id"]) > 100
                        or not isinstance(card.get("name"), str) or not card["name"] or len(card["name"]) > 200):
                    raise ValueError("invalid_cardcorpus_card_record")
                if card["id"] in seen_cards:
                    raise ValueError("duplicate_cardcorpus_printing_id")
                seen_cards.add(card["id"])
                external_id = external_set + "/" + card["id"]
                if len(external_id) > 100:
                    raise ValueError("cardcorpus_printing_id_too_long")
                collector = card.get("number")
                if collector is not None and not isinstance(collector, dict):
                    raise ValueError("invalid_cardcorpus_number")
                numerator = collector.get("numerator") if collector else None
                if numerator is not None and (not isinstance(numerator, str) or len(numerator) > 40):
                    raise ValueError("invalid_cardcorpus_number")
                number_is_printed = isinstance(numerator, str) and bool(numerator)
                number = numerator if number_is_printed else ""
                language = card.get("language", data.get("language", "en"))
                if not isinstance(language, str) or not LANGUAGE.fullmatch(language) or len(language) > 10:
                    raise ValueError("invalid_cardcorpus_language")
                canonical_id = identity(external_set, card["id"], language)
                provider_id = "cc-" + key(external_id)[:32]
                picture = card.get("picture")
                if picture is not None and not isinstance(picture, dict):
                    raise ValueError("invalid_cardcorpus_picture")
                image_data = picture.get("front") if isinstance(picture, dict) else None
                image_id = image_data.get("image") if isinstance(image_data, dict) else None
                image_url = _picture_url(index, image_id)
                variants = card.get("variants", [])
                if not isinstance(variants, list):
                    raise ValueError("invalid_cardcorpus_variants")
                options = []
                source_variants = []
                seen_variants = set()
                for variant in variants:
                    if (not isinstance(variant, dict) or not isinstance(variant.get("id"), str)
                            or not CARD_ID.fullmatch(variant["id"]) or len(variant["id"]) > 180):
                        raise ValueError("invalid_cardcorpus_variant")
                    if variant["id"] in seen_variants:
                        raise ValueError("duplicate_cardcorpus_variant_id")
                    seen_variants.add(variant["id"])
                    finish, name = variant.get("finish"), variant.get("name")
                    if finish is not None and not isinstance(finish, str):
                        raise ValueError("invalid_cardcorpus_variant_finish")
                    if name is not None and not isinstance(name, str):
                        raise ValueError("invalid_cardcorpus_variant_name")
                    raw_variant = _compact_variant(variant)
                    marketplace_options = _market_options(market, card["id"], variant["id"])
                    if marketplace_options:
                        raw_variant["marketplaces"] = marketplace_options
                    stamp = variant.get("stamp")
                    option = {"variantId": variant["id"], "type": finish or "unknown",
                              "subtype": name or variant["id"], "marketplaces": marketplace_options}
                    if stamp:
                        option["stamp"] = [stamp] if isinstance(stamp, str) else stamp
                    options.append(option)
                    source_variants.append(raw_variant)
                original = _original(card, source_variants)
                prepared.append({"card": card, "external_id": external_id, "provider_id": provider_id,
                                 "canonical_id": canonical_id, "number": number, "number_is_printed": number_is_printed,
                                 "language": language, "image_id": image_id, "image_url": image_url,
                                 "options": options, "original": original,
                                 "painting": (card.get("links") or {}).get("painting")})

            external_ids = [item["external_id"] for item in prepared]
            provider_ids = [item["provider_id"] for item in prepared]
            canonical_ids = [item["canonical_id"] for item in prepared]
            with Session.begin() as db:
                source_mappings = db.scalars(select(Mapping).where(
                    Mapping.provider == "cardcorpus", Mapping.variant == "", Mapping.provider_id.in_(external_ids),
                )).all() if external_ids else []
                mappings_by_external = {}
                for mapping in source_mappings:
                    mappings_by_external.setdefault(mapping.provider_id, []).append(mapping)
                    if len({entry.card_id for entry in mappings_by_external[mapping.provider_id]}) > 1:
                        raise ValueError("ambiguous_cardcorpus_mapping")
                mapped_ids = {mapping.card_id for mapping in source_mappings if mapping.card_id}
                mapped_cards = {row.id: row for row in db.scalars(select(Card).where(Card.id.in_(mapped_ids))).all()} if mapped_ids else {}
                existing_by_provider = {row.provider_id: row for row in db.scalars(
                    select(Card).where(Card.provider_id.in_(provider_ids))).all()} if provider_ids else {}
                existing_by_identity = {row.id: row for row in db.scalars(
                    select(Card).where(Card.id.in_(canonical_ids))).all()} if canonical_ids else {}
                new_rows = []
                pending_mappings = []
                for item in prepared:
                    card = item["card"]
                    source_mapping = mappings_by_external.get(item["external_id"], [None])[0]
                    row = (mapped_cards.get(source_mapping.card_id) if source_mapping and source_mapping.card_id else None)
                    row = row or existing_by_provider.get(item["provider_id"])
                    if row is None:
                        alias = existing_by_identity.get(item["canonical_id"])
                        if alias and _same_name(alias.name, card["name"]):
                            row = alias
                    if row is not None:
                        cid = row.id
                        details = dict(row.details or {})
                        prior_corpus = details.get("cardcorpus")
                        prior_corpus = prior_corpus if isinstance(prior_corpus, dict) else {}
                        prior_id = details.get("cardcorpus_id") or prior_corpus.get("id")
                        if prior_id not in (None, item["external_id"]):
                            raise ValueError("cardcorpus_identity_conflict")
                        local_variant_ids = {variant.get("variantId") for variant in details.get("variants_detailed", [])
                                             if isinstance(variant, dict)}
                        source_variant_options = [
                            {"id": option["variantId"], "type": option["type"], "subtype": option["subtype"]}
                            for option in item["options"] if option["variantId"] not in local_variant_ids]
                        new_corpus = {"id": item["external_id"], "sha256": source_hash,
                                      "picture_id": item["image_id"], "image_url": item["image_url"],
                                      "number_is_printed": item["number_is_printed"],
                                      "variants": item["original"].get("variants", []),
                                      "source_variant_options": source_variant_options,
                                      "original": item["original"]}
                        if item["painting"] is not None:
                            new_corpus["painting"] = item["painting"]
                        details["cardcorpus"] = {**_merge_missing(new_corpus, prior_corpus),
                                                  "source_variant_options": source_variant_options}
                        details.setdefault("cardcorpus_id", item["external_id"])
                        details.setdefault("number_is_printed", item["number_is_printed"])
                        if not details.get("image_url") and item["image_url"]:
                            details["image_url"] = item["image_url"]
                        if details.get("source") == "CardCorpus":
                            if not details.get("rarity") and card.get("rarity"):
                                details["rarity"] = card["rarity"]
                            if not details.get("set"):
                                set_id = SET_ALIASES.get(external_set, "cc-" + external_set.replace("/", "-"))
                                details["set"] = {"id": set_id, "name": data["name"],
                                                  "cardCount": {"official": data.get("printedTotal")}}
                        row.details = details
                        if not row.image and item["image_url"]:
                            row.image = item["image_url"]
                        linked += 1
                    else:
                        set_id = SET_ALIASES.get(external_set, "cc-" + external_set.replace("/", "-"))
                        details = {"source": "CardCorpus", "source_at": now(), "source_sha256": source_hash,
                                   "cardcorpus_id": item["external_id"], "image_url": item["image_url"],
                                   "picture_id": item["image_id"], "number_is_printed": item["number_is_printed"],
                                   "rarity": card.get("rarity") or "",
                                   "set": {"id": set_id, "name": data["name"],
                                           "cardCount": {"official": data.get("printedTotal")}},
                                   "variants_detailed": item["options"], "original": item["original"],
                                   "cardcorpus": {"id": item["external_id"], "sha256": source_hash,
                                                  "picture_id": item["image_id"], "image_url": item["image_url"],
                                                  "number_is_printed": item["number_is_printed"],
                                                  "variants": item["original"].get("variants", []),
                                                  "source_variant_options": [],
                                                  "original": item["original"], **({"painting": item["painting"]} if item["painting"] else {})}}
                        row = Card(id=item["canonical_id"], provider_id=item["provider_id"], set_id=set_id,
                                   number=item["number"], language=item["language"], name=card["name"],
                                   image=item["image_url"], details=details)
                        db.add(row)
                        new_rows.append(row)
                        existing_by_provider[item["provider_id"]] = row
                        existing_by_identity[item["canonical_id"]] = row
                        saved += 1
                        cid = row.id
                    if source_mapping is None:
                        pending_mappings.append(Mapping(id=key("cardcorpus:" + item["external_id"]),
                                                        card_id=cid, provider="cardcorpus",
                                                        provider_id=item["external_id"], variant="", status="matched"))
                    elif source_mapping.card_id is None:
                        source_mapping.card_id = cid
                    summary["card_records"] += 1
                    summary["distinct_printings"] += 1
                    summary["variant_options"] += len(item["options"])
                    summary["front_image_records"] += bool(item["image_id"])
                    summary["missing_front_image_records"] += not bool(item["image_id"])
                if new_rows:
                    db.flush()
                if pending_mappings:
                    db.add_all(pending_mappings)
            by_catalog[catalog] = summary
            log.info("cardcorpus_set_imported set=%s count=%d", external_set, len(prepared))

    for summary in by_catalog.values():
        for field in totals:
            totals[field] += summary[field]

    audit = {"source": "CardCorpus", "version": index.get("version"),
             "snapshot_sha256": snapshot_hash.hexdigest(), "catalogs": by_catalog,
             "totals": totals}
    return {"added": saved, "linked": linked, **totals, "source": "CardCorpus",
            "version": index.get("version"), "snapshot_sha256": audit["snapshot_sha256"],
            "catalogs": by_catalog, "audit": audit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=os.getenv("CARDCORPUS_DIR", "data/cardcorpus-snapshot/v1"))
    parser.add_argument("--catalog", action="append", choices=sorted(CATALOGS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(import_snapshot(args.root, args.catalog or ("international", "japanese", "zh-hans")), indent=2))
