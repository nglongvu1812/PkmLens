"""Pair the pinned CardCorpus gallery with printing IDs in the PkmLens catalog."""
import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from sqlalchemy import select

from cardlens_ai.identity import contained_path, sha256
from cardlens_ai.large_gallery import MODEL_REVISION, UPSTREAM_SHA256
from cardlens_core.catalog import variants
from cardlens_core.db import DATA, Card, Mapping, Session

UPSTREAM_SUMS_SHA256 = 'bfd2dfe652468605f8db1bd960f4e8c96ff6a18f184c2b599ace0c0f49769d09'
RUNTIME_FILES = tuple(UPSTREAM_SHA256)
CATALOGS = {'international', 'japanese', 'zh-hans'}


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        try:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.close()
            os.replace(stream.name, path)
        finally:
            Path(stream.name).unlink(missing_ok=True)


def _source_printings(snapshot):
    index = contained_path(snapshot, 'index.json').read_bytes()
    metadata = json.loads(index)
    if metadata.get('name') != 'CardCorpus' or not set(metadata.get('catalogs', ())).issubset(CATALOGS):
        raise ValueError('invalid_cardcorpus_snapshot')
    fingerprint = hashlib.sha256(index)
    rows = {}
    for catalog in sorted(metadata['catalogs']):
        folder = contained_path(snapshot, 'sets/' + catalog)
        for candidate in sorted(folder.glob('*.json')):
            path = contained_path(folder, candidate.name)
            data = path.read_bytes()
            fingerprint.update(path.relative_to(snapshot).as_posix().encode('utf-8') + b'\0' + hashlib.sha256(data).digest())
            record = json.loads(data)
            if record['id'] != f'{catalog}/{path.stem}':
                raise ValueError('invalid_cardcorpus_set_id')
            code = record.get('code')
            codes = [code.upper()] if isinstance(code, str) and re.fullmatch(r'[A-Za-z0-9]{2,12}', code) else []
            for card in record['cards']:
                external_id = record['id'] + '/' + card['id']
                if external_id in rows:
                    raise ValueError('duplicate_cardcorpus_printing_id')
                collector = card.get('number') or {}
                numerator = collector.get('numerator')
                printed = collector.get('printed') or ''
                denominator = re.search(r'/\s*([0-9]{1,4})$', printed)
                rows[external_id] = {'number': str(numerator) if numerator not in (None, '') else None,
                                     'number_is_printed': numerator not in (None, ''),
                                     'official_total': denominator[1] if denominator else None,
                                     'set_codes': codes,
                                     'source_variant_options': list(dict.fromkeys(v['id'] for v in card.get('variants', []) if v.get('id')))}
    if not rows:
        raise ValueError('empty_cardcorpus_snapshot')
    return rows, fingerprint.hexdigest()


def prepare_bundle(snapshot, bundle):
    snapshot, bundle = Path(snapshot).resolve(strict=True), Path(bundle).resolve(strict=True)
    sums = contained_path(bundle, 'SHA256SUMS')
    if sha256(sums) != UPSTREAM_SUMS_SHA256:
        raise ValueError('cardcorpus_release_checksum_mismatch')
    expected = {name: digest for digest, name in (line.split() for line in sums.read_text().splitlines() if line.strip())}
    hashes = {name: sha256(contained_path(bundle, name)) for name in RUNTIME_FILES}
    if any(digest != expected.get(name) for name, digest in hashes.items()):
        raise ValueError('cardcorpus_runtime_checksum_mismatch')
    source_rows, snapshot_hash = _source_printings(snapshot)
    printings = {}
    local_ids = set()
    statement = select(Mapping.provider_id, Card).join(Card, Card.id == Mapping.card_id).where(
        Mapping.provider == 'cardcorpus', Mapping.variant == '', Mapping.status == 'matched',
    ).order_by(Mapping.provider_id)
    with Session() as db:
        for external_id, card in db.execute(statement).yield_per(512):
            source = source_rows.get(external_id)
            if source is None:
                continue
            if external_id in printings or card.id in local_ids:
                raise ValueError('cardcorpus_mapping_not_one_to_one')
            local_ids.add(card.id)
            local_options = list(dict.fromkeys(v['id'] for v in variants(card.details)))
            source_options = [option for option in source.get('source_variant_options', []) if option not in local_options]
            printings[external_id] = {**source, 'card_id': card.id, 'set_id': card.set_id,
                'name': card.name, 'set': card.details.get('set', {}).get('name') or card.set_id,
                'language': card.language, 'variant_options': local_options,
                'source_variant_options': source_options}
    if not printings:
        raise ValueError('import_cardcorpus_metadata_first')
    catalog_variants = sum(len(row['variant_options']) + len(row['source_variant_options'])
                           for row in printings.values())
    mapping = {'schema_version': 1, 'snapshot_sha256': snapshot_hash,
               'catalog_variants': catalog_variants, 'printings': printings}
    mapping_path = contained_path(bundle, 'mapping.json')
    _atomic_json(mapping_path, mapping)
    hashes['mapping.json'] = sha256(mapping_path)
    _atomic_json(contained_path(bundle, 'checksums.json'), hashes)
    return {'mapped_printings': len(printings), 'snapshot_printings': len(source_rows),
            'catalog_variants': catalog_variants, 'snapshot_sha256': snapshot_hash,
            'mapping_sha256': hashes['mapping.json'], 'matcher_revision': MODEL_REVISION}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=DATA / 'cardcorpus-snapshot/v1')
    parser.add_argument('--bundle', type=Path, default=DATA / 'models/cardcorpus-scanner')
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(args.snapshot, args.bundle), indent=2))
