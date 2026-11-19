import json
from sqlalchemy import select, func
from cardlens_core.cardcorpus import import_snapshot, identity
from cardlens_core.db import Session, Card


def test_cardcorpus_import_is_idempotent_and_preserves_number_and_language(tmp_path):
    (tmp_path / 'sets/international').mkdir(parents=True)
    (tmp_path / 'index.json').write_text(json.dumps({'name': 'CardCorpus', 'version': '2.0.0',
        'catalogs': ['international'], 'pictures': {'sized': 'https://img.cardcorpus.com/s1/{width}/{id}.avif'}}))
    (tmp_path / 'sets/international/example.json').write_text(json.dumps({'id': 'international/example',
        'name': 'Example', 'language': 'en', 'cards': [{'id': '007', 'name': 'Card',
        'number': {'numerator': '007'}, 'variants': [{'id': 'holo', 'finish': 'holo'}]}]}))
    import_snapshot(tmp_path, ['international'])
    import_snapshot(tmp_path, ['international'])
    with Session() as db:
        assert db.scalar(select(func.count()).select_from(Card)) == 1
        card = db.scalar(select(Card))
        assert card.number == '007' and card.language == 'en'
        assert card.details['source'] == 'CardCorpus'
    assert identity('international/base-set', '7', 'en') == 'base1:7:en'


def test_distinct_printings_with_same_number_do_not_merge(tmp_path):
    (tmp_path / 'sets/international').mkdir(parents=True)
    (tmp_path / 'index.json').write_text(json.dumps({'name':'CardCorpus','version':'2.0.0',
        'catalogs':['international'],'pictures':{'sized':'https://img.cardcorpus.com/s1/{width}/{id}.avif'}}))
    cards = [{'id':key,'name':'Card','number':{'numerator':'007'},'variants':[]} for key in ['007','007-alt']]
    (tmp_path/'sets/international/example.json').write_text(json.dumps({'id':'international/example',
        'name':'Example','language':'en','cards':cards}))
    import_snapshot(tmp_path,['international'])
    import_snapshot(tmp_path,['international'])
    with Session() as db:
        assert db.scalar(select(func.count()).select_from(Card)) == 2
