import json

import pytest

from cardlens_core.catalog import now
from cardlens_core.cardcorpus import import_snapshot
from cardlens_core.db import Card, Identity, Listing, Mapping, Price, Quote, Scan, Session


def write_snapshot(root, cards_by_set, catalog="international", language="en"):
    folder = root / "sets" / catalog
    folder.mkdir(parents=True)
    (root / "index.json").write_text(json.dumps({
        "name": "CardCorpus", "version": "test", "catalogs": [catalog],
        "pictures": {"sized": "https://img.cardcorpus.com/s1/{width}/{id}.avif"},
    }), encoding="utf-8")
    for set_name, cards in cards_by_set.items():
        (folder / (set_name + ".json")).write_text(json.dumps({
            "id": catalog + "/" + ("base-set" if set_name == "base-set" else set_name),
            "name": set_name, "language": language,
            "printedTotal": 100, "cards": cards,
        }), encoding="utf-8")


def test_prepare_does_not_replace_existing_cardcorpus_printings(tmp_path, monkeypatch):
    import json
    import scripts.prepare as prepare
    from cardlens_core.db import Card, Session

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    catalog = tmp_path / "data" / "catalog"
    catalog.mkdir(parents=True)
    cached = {"detail": {"id": "source-provider", "set": {"id": "base1"},
                          "localId": "4", "name": "TCGdex copy", "variants_detailed": []},
              "fetched_at": "now"}
    (catalog / "source-provider.json").write_text(json.dumps(cached), encoding="utf-8")
    (tmp_path / "data" / "models").mkdir(parents=True)
    monkeypatch.setattr(prepare, "export_metadata", lambda: None)
    with Session.begin() as db:
        db.add(Card(id="base1:4:en", provider_id="source-provider", set_id="base1", number="4",
                    language="en", name="Charizard", image="", details={"source": "CardCorpus",
                    "cardcorpus_id": "international/base-set/4", "cardcorpus": {"id": "international/base-set/4"}}))

    with pytest.raises((FileNotFoundError, ValueError, KeyError)):
        prepare.prepare()

    with Session() as db:
        card = db.get(Card, "base1:4:en")
        assert card.name == "Charizard"
        assert card.details["source"] == "CardCorpus"
        assert card.details["cardcorpus"]["id"] == "international/base-set/4"

def test_import_counts_printings_variants_images_and_external_mappings(tmp_path):
    write_snapshot(tmp_path, {"set-a": [
        {"id": "007", "name": "Card A", "number": {"numerator": "7"},
         "picture": {"front": {"image": "a" * 32}},
         "variants": [{"id": "normal", "finish": "normal"}, {"id": "holo", "finish": "holo"}]},
        {"id": "007-alt", "name": "Card A", "number": {"numerator": "7"},
         "variants": [{"id": "reverse", "finish": "reverse"}]},
        {"id": "008", "name": "Card B", "number": {"numerator": "8"}, "variants": []},
    ]})

    result = import_snapshot(tmp_path, ["international"])
    assert result["card_records"] == result["distinct_printings"] == 3
    assert result["variant_options"] == 3 and result["front_image_records"] == 1
    with Session() as db:
        rows = db.query(Card).order_by(Card.id).all()
        assert len(rows) == 3
        card = next(row for row in rows if row.details["cardcorpus_id"].endswith("/007"))
        assert card.number == "7"
        assert [v["variantId"] for v in card.details["variants_detailed"]] == ["normal", "holo"]
        assert card.details["image_url"] == "https://img.cardcorpus.com/s1/480/" + "a" * 32 + ".avif"
        missing = next(row for row in rows if row.details["cardcorpus_id"].endswith("/008"))
        assert missing.image == "" and missing.details["image_url"] == ""
        mappings = db.query(Mapping).filter_by(provider="cardcorpus", variant="").all()
        assert {row.provider_id: row.card_id for row in mappings} == {
            "international/set-a/007": card.id,
            "international/set-a/007-alt": next(row.id for row in rows if row.id != card.id and row.number == "7"),
            "international/set-a/008": missing.id,
        }


def test_reimport_preserves_authoritative_variants_prices_listings_and_mapping_status(tmp_path):
    write_snapshot(tmp_path, {"set-a": [{
        "id": "007", "name": "Card A", "number": {"numerator": "007"},
        "variants": [{"id": "normal", "finish": "normal"}],
    }]})
    import_snapshot(tmp_path, ["international"])
    fetched = now()
    official_variant = {
        "variantId": "tcgdex-normal", "type": "normal", "subtype": "unlimited",
        "thirdParty": {"tcgplayer": "12345"},
        "pricing": {"tcgplayer": {"unit": "USD", "updated": fetched, "normal": {"marketPrice": 10}}},
    }
    with Session.begin() as db:
        card = db.query(Card).one()
        card.details = {**card.details, "variants_detailed": [official_variant],
                        "source_price_note": "keep", "original": {"id": "tcgdex-original"}}
        corpus_mapping = db.query(Mapping).filter_by(
            provider="cardcorpus", provider_id="international/set-a/007", variant="",
        ).one()
        corpus_mapping.status = "unmatched"
        db.add(Identity(id="owner", created_at=fetched, expires_at=fetched))
        db.add(Mapping(id="official-map", card_id=card.id, provider="tcgplayer", provider_id="12345",
                       variant="tcgdex-normal", status="unmatched"))
        db.add(Price(id="official-price", card_id=card.id, variant="tcgdex-normal", source="TCGdex/tcgplayer",
                     source_at=fetched, fetched_at=fetched, currency="USD", amount="10", low=None, high=None,
                     evidence={"mapping_status": "unmatched"}))
        db.add(Scan(id="scan", owner="owner", created_at=fetched, result={}, card_id=card.id,
                    variant="tcgdex-normal", condition="NM", revision=1))
        db.flush()
        db.add(Quote(id="quote", scan_id="scan", owner="owner", revision=1,
                     created_at=fetched, payload={"amount_vnd": 10}))
        db.flush()
        db.add(Listing(id="listing", owner="owner", scan_id="scan", quote_id="quote", card_id=card.id,
                       variant="tcgdex-normal", condition="NM", asking_vnd=10, status="draft",
                       created_at=fetched, details={"keep": "me"}))

    result = import_snapshot(tmp_path, ["international"])
    assert result["added"] == 0 and result["variant_options"] == 1
    with Session() as db:
        card = db.query(Card).one()
        assert card.details["variants_detailed"] == [official_variant]
        assert card.details["original"] == {"id": "tcgdex-original"}
        assert card.details["source_price_note"] == "keep"
        assert card.details["cardcorpus"]["variants"][0]["id"] == "normal"
        assert card.details["cardcorpus"]["source_variant_options"] == [
            {"id": "normal", "type": "normal", "subtype": "normal"},
        ]
        assert db.query(Mapping).filter_by(
            provider="cardcorpus", provider_id="international/set-a/007", variant="",
        ).one().status == "unmatched"
        assert db.get(Mapping, "official-map").status == "unmatched"
        assert db.get(Price, "official-price").amount == "10"
        assert db.get(Quote, "quote").payload == {"amount_vnd": 10}
        assert db.get(Listing, "listing").details == {"keep": "me"}


def test_source_language_is_exact_and_absent_print_number_is_not_guessed(tmp_path):
    write_snapshot(tmp_path, {"set-a": [{"id": "promo-1", "name": "Card", "language": "zh-Hans", "variants": []}]},
                   catalog="zh-hans", language="zh-Hant")
    import_snapshot(tmp_path, ["zh-hans"])
    with Session() as db:
        card = db.query(Card).one()
        assert card.id.endswith(":zh-Hans") and card.language == "zh-Hans"
        assert card.number == "" and card.details["number_is_printed"] is False
        assert "number" not in card.details["original"]
        assert card.details["variants_detailed"] == []


def test_only_known_legacy_nidoran_spacing_alias_enriches_existing_identity(tmp_path):
    write_snapshot(tmp_path, {"base-set": [{"id": "55", "name": "Nidoran " + chr(0x2642),
                   "number": {"numerator": "55"}, "variants": [{"id": "normal", "finish": "normal"}]}]})
    with Session.begin() as db:
        db.add(Card(id="base1:55:en", provider_id="base1-55", set_id="base1", number="55", language="en",
                    name="Nidoran" + chr(0x2642), image="", details={"source": "TCGdex",
                    "variants_detailed": [{"variantId": "authoritative"}]}))
    import_snapshot(tmp_path, ["international"])
    with Session() as db:
        card = db.get(Card, "base1:55:en")
        assert card.name == "Nidoran" + chr(0x2642)
        assert card.details["variants_detailed"] == [{"variantId": "authoritative"}]
        assert card.details["cardcorpus"]["id"] == "international/base-set/55"
        assert db.query(Mapping).filter_by(provider_id="international/base-set/55", provider="cardcorpus").one().card_id == card.id


def test_invalid_hosts_paths_and_picture_ids_are_rejected(tmp_path):
    write_snapshot(tmp_path, {"set-a": [{"id": "007", "name": "Card", "number": {"numerator": "7"},
                   "picture": {"front": {"image": "..\\\\private"}}, "variants": []}]})
    with pytest.raises(ValueError, match="picture_id"):
        import_snapshot(tmp_path, ["international"])
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    index["pictures"]["sized"] = "https://attacker.invalid/{width}/{id}"
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="picture_template"):
        import_snapshot(tmp_path, ["international"])
    with pytest.raises(ValueError, match="snapshot_root"):
        import_snapshot(tmp_path / "..", ["international"])


def test_import_totals_are_not_recounted_and_snapshot_hash_matches_bundle_preparer(tmp_path):
    from scripts.prepare_cardcorpus_gallery import _source_printings

    write_snapshot(tmp_path, {
        "set-a": [{"id": "1", "name": "One", "variants": [{"id": "normal"}]},
                  {"id": "2", "name": "Two", "variants": []}],
        "set-b": [{"id": "1", "name": "Three", "variants": [{"id": "normal"}, {"id": "holo"}]}],
    })

    result = import_snapshot(tmp_path, ["international"])
    _, prepared_fingerprint = _source_printings(tmp_path)

    assert result["card_records"] == result["distinct_printings"] == 3
    assert result["variant_options"] == 3
    assert result["catalogs"]["international"]["card_records"] == 3
    assert result["snapshot_sha256"] == prepared_fingerprint

def test_path_like_set_identity_and_duplicate_variant_ids_are_rejected(tmp_path):
    write_snapshot(tmp_path, {"set-a": [{"id": "007", "name": "Card",
                   "number": {"numerator": "7"}, "variants": [{"id": "normal"}, {"id": "normal"}]}]})
    with pytest.raises(ValueError, match="duplicate_cardcorpus_variant_id"):
        import_snapshot(tmp_path, ["international"])
    source_file = tmp_path / "sets/international/set-a.json"
    source = json.loads(source_file.read_text(encoding="utf-8"))
    source["id"] = "international/../../outside"
    source_file.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="set_record"):
        import_snapshot(tmp_path, ["international"])
