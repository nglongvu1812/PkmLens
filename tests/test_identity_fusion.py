"""Identity fusion unit fixtures only; no real-photo accuracy measurement."""
from dataclasses import replace
import numpy as np
from PIL import Image
import pytest
from cardlens_ai import pipeline
from cardlens_ai.recognition_config import DEFAULT_POLICY


def test_contradictory_number_abstains_and_keeps_candidates(monkeypatch):
    class Index:
        ntotal = 2
        def search(self, vector, count):
            return np.array([[.94, .70]], dtype='float32'), np.array([[0, 1]])
    rows = [{'card_id': 'a:007:en', 'number': '007'}, {'card_id': 'b:8:en', 'number': '8'}]
    monkeypatch.setattr(pipeline, 'gallery', lambda: (Index(), {'rows': rows, 'model': 'fixture', 'checkpoint_kind': 'unit_fixture', 'gallery_version': 'fixture'}, None))
    monkeypatch.setattr(pipeline, 'embedding', lambda image: np.ones((1, 2), dtype='float32'))
    monkeypatch.setattr(pipeline, 'read_number', lambda image: {'number': '7', 'total': None, 'status': 'read', 'roi_mode': 'fixture', 'text': ['7/100']})
    image = Image.fromarray(np.random.default_rng(1).integers(20, 230, (704, 504, 3), dtype=np.uint8))
    result = pipeline.identify(image)
    assert result['status'] == 'unknown'
    assert 'ocr_number_conflict' in result['reason_codes']
    assert result['raw_candidates'][0]['card_id'] == 'a:007:en'
    assert result['candidates'][0]['evidence']['number_match'] is False
    assert {'ocr', 'fusion', 'encoding', 'embedding', 'retrieval', 'quality'} <= result['timing_ms'].keys()


def test_name_conflict_requires_reliable_dissimilar_name():
    from cardlens_ai.identity import rank_candidates, decision
    rows = [{'card_id': 'a:7:en', 'number': '7', 'similarity': .95, 'name': 'Pikachu'}, {'card_id': 'b:8:en', 'similarity': .60, 'name': 'Squirtle'}]
    ocr = {'name': 'Charizard', 'name_confidence': .99}
    ranked = rank_candidates(rows, ocr)
    assert decision(ranked, ocr, DEFAULT_POLICY)['status'] == 'unknown'
    assert 'ocr_name_conflict' in decision(ranked, ocr, DEFAULT_POLICY)['reason_codes']
    uncertain = rank_candidates(rows, {'name': 'Charizard', 'name_confidence': .5})
    assert uncertain[0]['evidence']['name_match'] is None
    partial = rank_candidates(rows, {'name': 'Pikach', 'name_confidence': .99})
    assert partial[0]['evidence']['name_match'] is None
    assert 'evidence' not in rows[0]


def test_set_conflict_and_fused_margin_are_conservative():
    from cardlens_ai.identity import rank_candidates, decision
    rows = [{'card_id': 'a:7:en', 'similarity': .9, 'set_codes': ['SVI']}, {'card_id': 'b:8:en', 'similarity': .89, 'set_codes': ['OBF']}]
    ocr = {'number': '7', 'set_code': 'OBF', 'set_code_confidence': .99}
    ranked = rank_candidates(rows, ocr)
    result = decision(ranked, ocr, DEFAULT_POLICY)
    assert result['status'] == 'unknown'
    assert result['margin'] == pytest.approx(ranked[0]['fusion_score'] - ranked[1]['fusion_score'])
    ranked = rank_candidates(rows, {})
    assert 'ambiguous_margin' in decision(ranked, {}, DEFAULT_POLICY)['reason_codes']
    assert decision(ranked, {}, replace(DEFAULT_POLICY, min_margin=.001))['status'] == 'needs_confirmation'


def test_quality_gate_and_method_allowlist_precede_models(monkeypatch):
    monkeypatch.setattr(pipeline, 'gallery', lambda: pytest.fail('unusable image loaded gallery'))
    with pytest.raises(ValueError, match='invalid_identification_method'):
        pipeline.identify(Image.new('RGB', (64, 90)), method='typo')
    result = pipeline.identify(Image.new('RGB', (64, 90), 'gray'))
    assert result['status'] == 'unknown' and not result['quality']['usable']
    assert 'blurred' in result['reason_codes']


def test_catalog_metadata_does_not_invent_variant_language_or_set_code(tmp_path):
    import json
    from cardlens_ai.identity import catalog_metadata
    catalog = tmp_path / 'catalog'
    catalog.mkdir()
    (catalog / 'one.json').write_text(json.dumps({'detail': {'name': 'Pikachu', 'localId': '007', 'set': {'id': 'internal', 'name': 'Sample', 'cardCount': {'official': 100}}, 'variants_detailed': [{'variantId': 'holo', 'type': 'holo'}]}}))
    row = catalog_metadata({'provider_id': 'one', 'card_id': 'set:007:en'}, tmp_path)
    assert row['number'] == '007' and row['language'] == 'en'
    assert row['variant'] is None and row['set_codes'] == []
    with pytest.raises(ValueError, match='unsafe_catalog_provider'):
        catalog_metadata({'provider_id': '../outside', 'card_id': 'x:1:en'}, tmp_path)


def test_legacy_rerank_preserves_leading_zero_and_evidence():
    row = pipeline.rerank([{'card_id': 'a:007:en', 'similarity': .9}], '7', '102')[0]
    assert row['evidence']['number_match'] is False
    assert row['evidence']['catalog_number'] == '007'
    assert row['evidence']['ocr_total'] == '102'


def test_number_only_cannot_override_visual_top_candidate():
    from cardlens_ai.identity import rank_candidates, decision
    rows = [{'card_id': 'a:7:en', 'similarity': .95}, {'card_id': 'b:8:en', 'similarity': .90}]
    ocr = {'number': '8'}
    ranked = rank_candidates(rows, ocr)
    assert ranked[0]['card_id'] == 'b:8:en'  # Preserve legacy ranking for manual correction.
    result = decision(ranked, ocr, DEFAULT_POLICY)
    assert result['status'] == 'unknown' and 'number_only_override' in result['reason_codes']


def test_ocr_reads_confident_name_and_explicit_code_once_per_region(monkeypatch):
    outputs = iter([([[[], 'OBF EN 223/197', .98]], None),
                    ([[[[0, 0], [100, 0], [100, 10], [0, 10]], 'Evolvesfrom Charmander', .99],
                      [[[0, 12], [100, 12], [100, 22], [0, 22]], 'PutCharizard on the Stage I card', .99],
                      [[[0, 24], [100, 24], [100, 50], [0, 50]], 'Charizard ex', .95]], None)])
    monkeypatch.setattr(pipeline, 'ocr_engine', lambda: lambda image: next(outputs))
    result = pipeline.read_number(Image.new('RGB', (504, 704)))
    assert result['number'] == '223' and result['total'] == '197'
    assert result['name'] == 'Charizard ex' and result['name_confidence'] == .95
    assert result['set_code'] == 'OBF' and result['language'] == 'en'
    assert result['timing_ms']['ocr'] >= 0 and result['timing_ms']['encoding'] >= 0


def test_unsupported_language_abstains_without_variant_fabrication():
    from cardlens_ai.identity import rank_candidates, decision
    rows = [{'card_id': 'a:7:en', 'similarity': .95, 'language': 'en', 'variant': None},
            {'card_id': 'b:7:en', 'similarity': .7, 'language': 'en', 'variant': None}]
    ocr = {'language': 'ja', 'language_confidence': .99}
    ranked = rank_candidates(rows, ocr)
    result = decision(ranked, ocr, DEFAULT_POLICY)
    assert result['status'] == 'unknown' and 'ocr_language_conflict' in result['reason_codes']
    assert ranked[0]['variant'] is None


def test_single_candidate_is_not_a_measured_competitor_margin():
    from cardlens_ai.identity import rank_candidates, decision
    result = decision(rank_candidates([{'card_id': 'a:7:en', 'similarity': .99}], {}), {}, DEFAULT_POLICY)
    assert result['status'] == 'unknown' and result['margin'] is None
    assert 'missing_competitor' in result['reason_codes']


def test_matching_number_does_not_rescue_weak_visual_similarity():
    from cardlens_ai.identity import rank_candidates, decision
    ocr = {'number': '7'}
    ranked = rank_candidates([{'card_id': 'a:7:en', 'similarity': .65}, {'card_id': 'b:8:en', 'similarity': .6}], ocr)
    result = decision(ranked, ocr, DEFAULT_POLICY)
    assert result['status'] == 'unknown' and 'weak_image_similarity' in result['reason_codes']


def test_shared_veto_keeps_number_only_and_live_number_requirements_distinct():
    from cardlens_ai.identity import identity_vetoes, rank_candidates, decision
    ocr = {"number": "8"}
    ranked = rank_candidates([
        {"card_id": "a:7:en", "similarity": .95},
        {"card_id": "b:8:en", "similarity": .90},
    ], ocr)
    result = decision(ranked, ocr, DEFAULT_POLICY)
    assert result["status"] == "unknown"
    assert "number_only_override" in result["reason_codes"]
    assert "number_only_override" in identity_vetoes(
        ranked, ocr, DEFAULT_POLICY.open_set_min_similarity, raw_candidates=ranked,
        require_number_match=True)

    no_number = rank_candidates([
        {"card_id": "a:7:en", "similarity": .95},
        {"card_id": "b:8:en", "similarity": .60},
    ], {})
    assert decision(no_number, {}, DEFAULT_POLICY)["status"] == "needs_confirmation"
    assert "ocr_missing_or_ambiguous" in identity_vetoes(
        no_number, {}, DEFAULT_POLICY.open_set_min_similarity, raw_candidates=no_number,
        require_number_match=True)


def test_same_name_reprints_use_distinct_competitor_margin():
    from cardlens_ai.identity import rank_candidates, decision
    reprints = rank_candidates([
        {"card_id": "base1:54:en", "name": "Metapod", "similarity": .973},
        {"card_id": "lc:54:en", "name": "Metapod", "similarity": .957},
        {"card_id": "base1:10:en", "name": "Caterpie", "similarity": .720},
    ], {})
    result = decision(reprints, {}, DEFAULT_POLICY)
    assert result["status"] == "needs_confirmation"
    assert "ambiguous_margin" not in result["reason_codes"]
    assert result["margin"] == pytest.approx(reprints[0]["fusion_score"] - reprints[1]["fusion_score"])
