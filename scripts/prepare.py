"""Prepare independent PkmLens data from cached references and CardCorpus."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'services/core-api'), str(ROOT / 'services/ai'), str(ROOT)]
os.environ.setdefault('DATA_DIR', str(ROOT / 'data'))
os.environ.setdefault('DATABASE_URL', 'sqlite:///' + (ROOT / 'data/pkmlens.db').as_posix())

from cardlens_core.db import Base, engine, Session, Card, Fx
from cardlens_core.catalog import save_card, now, sync_fx


def prepare():
    Base.metadata.create_all(engine)
    data = ROOT / 'data'
    with Session() as db:
        imported = {provider_id for (provider_id,) in db.query(Card.provider_id)}
    for path in (data / 'catalog').glob('*.json'):
        cached = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(cached, dict) and 'detail' in cached and cached['detail']['id'] not in imported:
            provider_id = cached['detail']['id']
            save_card(cached['detail'], cached['fetched_at'])
            imported.add(provider_id)
    with Session() as db:
        has_fx = db.query(Fx).first() is not None
    if not has_fx:
        try:
            sync_fx()
        except Exception as exc:
            print('FX unavailable; prices will stay unavailable:', type(exc).__name__)
    result = json.loads((data / 'models/source-result.json').read_text(encoding='utf-8'))
    checkpoint = data / 'models/card-detector-best.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != result['sha256']:
        raise ValueError('detector_checksum_mismatch')
    registry_path = data / 'models/detectors.json'
    if not registry_path.exists():
        result.update(id='cardlens-synthetic-yolo26n', checkpoint=str(checkpoint),
                      provenance='Reused CardLens checkpoint; synthetic validation only')
        registry_path.write_text(json.dumps({'active': result['id'], 'models': [result]}, indent=2), encoding='utf-8')
    export_metadata()
    print('PkmLens database ready. No user accounts or listings copied.')


def export_metadata():
    from sqlalchemy import select
    with Session() as db, (ROOT / 'data/metadata.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_path', 'card_id', 'name', 'set', 'number', 'rarity'])
        for card in db.scalars(select(Card)):
            path = ROOT / 'data/gallery' / (card.provider_id + '.webp')
            if path.is_file():
                writer.writerow([path.relative_to(ROOT / 'data').as_posix(), card.id, card.name,
                                 card.details.get('set', {}).get('name', card.set_id), card.number,
                                 card.details.get('rarity', '')])


def download_cardcorpus():
    import httpx
    from PIL import Image
    from sqlalchemy import select
    snapshot = Path(os.getenv('CARDCORPUS_DIR', r'D:\Project\CardLens\cardcorpus-dataset-2026-11-12\v1'))
    selections = {
        'scarlet-violet-151': ['003', '006', '009', '025', '151', '166', '168', '170', '172', '173', '175', '198', '199', '200', '201', '202', '203', '205'],
        'obsidian-flames': ['125', '197', '199', '210', '215', '223'],
    }
    index = json.loads((snapshot / 'index.json').read_text(encoding='utf-8'))
    rows = []
    for set_name, wanted in selections.items():
        source = snapshot / 'sets/international' / (set_name + '.json')
        raw = json.loads(source.read_text(encoding='utf-8'))
        # Keep the small, attributed snapshot needed by this demo in PkmLens.
        cache = ROOT / 'data/cardcorpus'
        cache.mkdir(exist_ok=True)
        (cache / (set_name + '.json')).write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
        for card in raw['cards']:
            if card['id'] not in wanted:
                continue
            image_id = card.get('picture', {}).get('front', {}).get('image')
            if not image_id:
                continue
            provider_id = 'cc-' + hashlib.sha256((raw['id'] + '/' + card['id']).encode()).hexdigest()[:32]
            set_id = 'cc-' + raw['id'].replace('/', '-')
            cid = f'{set_id}:{card["id"]}:en'
            detail = {'id': provider_id, 'localId': card['id'], 'name': card['name'],
                      'source': 'CardCorpus', 'cardcorpus_id': raw['id'] + '/' + card['id'],
                      'rarity': card.get('rarity', ''), 'illustrator': card.get('credit', ''),
                      'set': {'id': set_id, 'name': raw['name'], 'cardCount': {'official': raw['printedTotal']}},
                      'variants_detailed': [{'variantId': v['id'], 'type': v.get('finish', ''), 'subtype': v.get('name', '')}
                                            for v in card.get('variants', [])],
                      'original': card}
            url = index['pictures']['sized'].format(width=480, id=image_id)
            rows.append((cid, provider_id, detail, url))

    def download(row):
        cid, provider_id, detail, url = row
        target = ROOT / 'data/gallery' / (provider_id + '.webp')
        if not target.exists():
            response = httpx.get(url, timeout=35, follow_redirects=True)
            response.raise_for_status()
            if len(response.content) > 8 * 1024 * 1024:
                raise ValueError('image_size_limit')
            image = Image.open(io.BytesIO(response.content))
            if image.width * image.height > 16_000_000:
                raise ValueError('image_dimensions_limit')
            image.convert('RGB').save(target, 'WEBP', quality=94)
        cached = {'detail': detail, 'fetched_at': now(), 'source': url}
        (ROOT / 'data/catalog' / (provider_id + '.json')).write_text(json.dumps(cached, ensure_ascii=False), encoding='utf-8')
        save_card(detail, cached['fetched_at'])
        return {'card_id': cid, 'provider_id': provider_id, 'source': url,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}

    completed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row, result in zip(rows, pool.map(download, rows)):
            completed.append(result)
            print('CardCorpus', row[2]['name'], row[2]['localId'], flush=True)
    (ROOT / 'data/catalog/cardcorpus-manifest.json').write_text(json.dumps(completed, indent=2), encoding='utf-8')
    export_metadata()
    return len(completed)


if __name__ == '__main__':
    prepare()
    if '--cardcorpus' in sys.argv:
        print('Downloaded', download_cardcorpus(), 'CardCorpus references')
    if '--index' in sys.argv:
        from cardlens_ai.pipeline import build_index
        print('Gallery vectors:', build_index()['count'])
