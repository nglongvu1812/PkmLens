from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cardlens_ai import identity, live, main, pipeline


def sharp_image():
    return Image.fromarray(np.random.default_rng(31).integers(20, 230, (704, 504, 3), dtype=np.uint8))


class PairedGallery:
    coverage = {'reference_images': 100000, 'indexed_printings': 67000, 'indexed_catalog_variants': 99000}
    model_versions = {'kind': 'external_pretrained_matcher', 'gallery': 'test-paired',
                      'identity_fingerprint': 'paired-only'}

    def search(self, image, top_k):
        assert image.size == (504, 704) and top_k == 50
        return {'candidates': [
            {'card_id': 'local:007:ja', 'number': '007', 'number_is_printed': True,
             'language': 'ja', 'variant': None, 'reference_variant': 'holo', 'similarity': .95},
            {'card_id': 'local:008:ja', 'number': '008', 'number_is_printed': True,
             'language': 'ja', 'variant': None, 'similarity': .81},
        ], 'timing_ms': {'embedding': 2, 'retrieval': 3}, 'model_versions': self.model_versions}


@pytest.mark.parametrize('observed,expected', [('007', 'needs_confirmation'), ('006', 'unknown')])
def test_paired_matcher_preserves_ocr_gate_and_never_verifies_reference_variant(monkeypatch, observed, expected):
    monkeypatch.setattr(pipeline, 'gallery', lambda: pytest.fail('used unrelated gallery'))
    monkeypatch.setattr(pipeline, 'embedding', lambda _: pytest.fail('used unrelated matcher'))
    monkeypatch.setattr(pipeline, 'read_number', lambda _: {'number': observed, 'roi_mode': 'fixture'})
    trace = pipeline.identify(sharp_image(), cardcorpus=PairedGallery())
    assert trace['status'] == expected
    assert trace['model_versions']['identity_fingerprint'] == 'paired-only'
    assert trace['model_versions']['gallery'] == 'test-paired'
    assert trace['candidates'][0]['card_id'] == 'local:007:ja'
    assert trace['candidates'][0]['variant'] is None
    assert not trace['candidates'][0]['evidence']['variant_verified']
    assert trace['timing_ms']['retrieval'] == 3


def test_absent_printed_number_is_not_supplied_by_catalog_id():
    rows = [{'card_id': 'set:123:en', 'number': None, 'number_is_printed': False, 'similarity': .91}]
    evidence = identity.rank_candidates(rows, {'number': '123'})[0]['evidence']
    assert evidence['catalog_number'] is None and evidence['number_match'] is None


def test_large_gallery_keeps_quality_gate(monkeypatch):
    never = SimpleNamespace(search=lambda *a, **k: pytest.fail('retrieved for a blurred image'))
    assert pipeline.identify(Image.new('RGB', (504, 704), 'gray'), cardcorpus=never)['status'] == 'unknown'


def test_prepared_bundle_is_not_reported_loaded_until_runtime_loads_it(tmp_path, monkeypatch):
    root = tmp_path / "models" / "cardcorpus-scanner"
    root.mkdir(parents=True)
    (root / "checksums.json").write_text("{}", encoding="utf-8")
    (root / "mapping.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setenv("PKMLENS_GALLERY_BACKEND", "auto")
    monkeypatch.setattr(pipeline, "active_cardcorpus_gallery", lambda: None)

    status = pipeline.cardcorpus_bundle_status()

    assert status == {"requested": "auto", "active": "legacy", "prepared": True,
                      "large_index_loaded": False, "available": True}

def test_live_and_http_use_same_configured_gallery(monkeypatch):
    gallery = PairedGallery()
    monkeypatch.setattr(pipeline, 'active_cardcorpus_gallery', lambda: gallery)
    monkeypatch.setattr(main, 'active_cardcorpus_gallery', lambda: gallery, raising=False)
    monkeypatch.setattr(pipeline, 'read_number', lambda _: {'number': '007', 'roi_mode': 'fixture'})
    monkeypatch.setattr(pipeline, 'gallery', lambda: pytest.fail('requested legacy gallery for a scan'))
    assert live.recognize(sharp_image())['model_versions']['gallery'] == 'test-paired'
    from io import BytesIO
    from fastapi.testclient import TestClient

    encoded = BytesIO()
    sharp_image().save(encoded, format='JPEG')
    client = TestClient(main.app)
    readiness = client.get('/readyz').json()
    assert readiness['gallery']['coverage']['indexed_printings'] == 67000
    assert readiness['gallery']['requested'] == 'auto'
    assert readiness['gallery']['active'] == 'cardcorpus'
    assert readiness['gallery']['large_index_loaded'] is True
    response = client.post('/v1/identify', files={'image': ('scan.jpg', encoded.getvalue(), 'image/jpeg')})
    assert response.status_code == 200
    assert response.json()['model_versions']['gallery'] == 'test-paired'
