import json
from pathlib import Path
from PIL import Image
from ml.detection_dataset import generate, split_for


def test_multi_card_labels_and_source_split_are_consistent(tmp_path):
    gallery = tmp_path / 'gallery'
    gallery.mkdir()
    for i in range(40):
        Image.new('RGB', (126,176), (i*5, 120, 70)).save(gallery / f'card-{i}.webp')
    output = tmp_path / 'prepared'
    info = generate(gallery, output, count=12)
    rows = [json.loads(s) for s in (output/'manifest.jsonl').read_text().splitlines()]
    assert info['real_photo_count'] == 0
    assert any(len(row['objects']) > 1 for row in rows)
    assert any(not row['objects'] for row in rows)
    seen = {}
    for row in rows:
        for obj in row['objects']:
            assert seen.setdefault(obj['group'], row['split']) == row['split']
        label = output/'labels'/row['split']/(row['sample_id']+'.txt')
        for line in label.read_text().splitlines():
            values = list(map(float, line.split()))
            assert len(values) == 5 and all(0 <= v <= 1 for v in values)
    assert split_for(Path('base1-7.webp')) == split_for(Path('base4-8.webp'))


def test_cardcorpus_shared_artwork_stays_in_one_source_group(tmp_path):
    from ml.detection_dataset import artwork_groups
    gallery = tmp_path / 'gallery'
    catalog = tmp_path / 'catalog'
    gallery.mkdir(); catalog.mkdir()
    for ident in ('first', 'reprint'):
        Image.new('RGB', (90, 130)).save(gallery / (ident + '.webp'))
        (catalog / (ident + '.json')).write_text(json.dumps({'detail': {'original': {
            'links': {'painting': {'set': 'japanese/source', 'painting': '025'}}
        }}}))
    result = artwork_groups(gallery)
    assert result['first'] == result['reprint']
