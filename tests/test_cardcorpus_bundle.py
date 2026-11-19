import hashlib
import json
from pathlib import Path

import pytest

from cardlens_core.db import Card, Mapping, Session


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_bundle_maps_existing_card_id_and_only_sourced_number(tmp_path, monkeypatch):
    from scripts import prepare_cardcorpus_gallery as prepare

    snapshot = tmp_path / 'snapshot'
    folder = snapshot / 'sets/international'
    folder.mkdir(parents=True)
    (snapshot / 'index.json').write_text(json.dumps({'name': 'CardCorpus', 'catalogs': ['international']}))
    (folder / 'sample.json').write_text(json.dumps({'id': 'international/sample', 'code': 'AA1', 'cards': [
        {'id': 'a-alt', 'number': {'numerator': '007', 'printed': '007/099'}}, {'id': 'energy'},
    ]}))
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'pipeline.json').write_text('paired model')
    sums = bundle / 'SHA256SUMS'
    sums.write_text(digest(bundle / 'pipeline.json') + '  pipeline.json\n')
    monkeypatch.setattr(prepare, 'UPSTREAM_SUMS_SHA256', digest(sums))
    monkeypatch.setattr(prepare, 'RUNTIME_FILES', ('pipeline.json',))
    with Session.begin() as db:
        for external, cid, number, options in [('a-alt', 'legacy:7:en', '7', ['holo', 'holo', 'reverse']),
                                               ('energy', 'legacy:e:en', 'energy', ['normal'])]:
            db.add(Card(id=cid, provider_id='p-' + external, set_id='legacy', number=number, language='en',
                        name=external, image='', details={'variants_detailed': [{'variantId': v} for v in options]}))
        db.flush()
        for external, cid in [('a-alt', 'legacy:7:en'), ('energy', 'legacy:e:en')]:
            db.add(Mapping(id=external, card_id=cid, provider='cardcorpus',
                           provider_id='international/sample/' + external, variant='', status='matched'))
    summary = prepare.prepare_bundle(snapshot, bundle)
    mapping = json.loads((bundle / 'mapping.json').read_bytes())
    assert summary['catalog_variants'] == 3 and summary['mapped_printings'] == 2
    card = mapping['printings']['international/sample/a-alt']
    assert card['card_id'] == 'legacy:7:en' and card['number'] == '007'
    assert card['official_total'] == '099' and card['set_codes'] == ['AA1']
    assert card['variant_options'] == ['holo', 'reverse']
    assert mapping['printings']['international/sample/energy']['number'] is None
    assert json.loads((bundle / 'checksums.json').read_bytes())['mapping.json'] == digest(bundle / 'mapping.json')
    assert prepare.prepare_bundle(snapshot, bundle) == summary
    (bundle / 'pipeline.json').write_text('wrong model')
    with pytest.raises(ValueError, match='checksum'):
        prepare.prepare_bundle(snapshot, bundle)
