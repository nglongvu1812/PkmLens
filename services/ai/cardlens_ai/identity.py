"""Conservative, uncalibrated OCR evidence; identity scores are not probabilities."""
import hashlib
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

FUSION_VERSION = 'ocr-identity-v3'
PROJECTION_KIND = 'fine_tuned_projection_frozen_backbone'
CALIBRATION_RULE = 'live_frame_ocr_consistent_fusion_margin_v2'
NON_TUNED_POLICY_FIELDS = ('open_set_min_similarity', 'min_sharpness', 'min_brightness', 'max_glare')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('path_outside_allowed_root')
    return path


def catalog_metadata(row, data):
    """Enrich local records only. A catalog variant option is not a verified variant."""
    result = dict(row)
    provider = row.get('provider_id', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', provider):
        raise ValueError('unsafe_catalog_provider')
    path = contained_path(Path(data) / 'catalog', provider + '.json')
    detail = json.loads(path.read_text(encoding='utf-8'))['detail'] if path.is_file() else {}
    card_set = detail.get('set') or {}
    parts = row['card_id'].split(':')
    original = detail.get('original') or {}
    codes = [card_set.get(key) for key in ('code', 'tcgOnline', 'abbreviation')]
    # Internal catalog IDs and set names need not be the printed set code.
    result.update(name=detail.get('name', row.get('name')),
                  set=card_set.get('name', row.get('set')),
                  set_id=card_set.get('id', row.get('set_id')),
                  number=str(detail.get('localId') or row.get('number') or (parts[1] if len(parts) > 1 else '')) or None,
                  language=detail.get('language') or original.get('language') or row.get('language') or (parts[2] if len(parts) > 2 and parts[2] else None),
                  variant=None,
                  set_codes=list(dict.fromkeys(str(code).upper() for code in codes if isinstance(code, str) and code)),
                  official_total=str(card_set['cardCount']['official']) if (card_set.get('cardCount') or {}).get('official') else row.get('official_total'))
    result['variant_options'] = [{key: variant[key] for key in ('variantId', 'type', 'subtype', 'stamp', 'size') if key in variant}
                                 for variant in detail.get('variants_detailed', [])]
    return result


def _name(value):
    value = unicodedata.normalize('NFKD', value or '')
    return ''.join(c.lower() for c in value if c.isalnum())


def _reliable(ocr, key):
    confidence = ocr.get(key + '_confidence')
    return (isinstance(confidence, (float, int)) and not isinstance(confidence, bool)
            and math.isfinite(confidence) and .85 <= confidence <= 1)


def rank_candidates(candidates, ocr):
    """Keep legacy number/total evidence and exact collector-number formatting."""
    ranked = []
    for source in candidates:
        candidate = dict(source)
        number, total = ocr.get('number'), ocr.get('total')
        parts = candidate['card_id'].split(':')
        expected = candidate.get('number') if candidate.get('number_is_printed') is not False else None
        if expected in (None, '') and candidate.get('number_is_printed') is not False:
            expected = parts[1] if len(parts) > 1 else None
        catalog_total = candidate.get('official_total')
        number_match = None if number is None or expected is None else str(expected).upper() == str(number).upper()
        total_match = None if total is None or catalog_total is None else str(total) == str(catalog_total)
        observed, name = _name(ocr.get('name')), _name(candidate.get('name'))
        name_match = None
        if observed and name and _reliable(ocr, 'name'):
            if observed == name:
                name_match = True
            elif (min(len(observed), len(name)) >= 4 and observed not in name and name not in observed
                  and SequenceMatcher(None, observed, name).ratio() < .55):
                name_match = False
        codes = candidate.get('set_codes') or []
        code_match = None
        if codes and ocr.get('set_code') and _reliable(ocr, 'set_code'):
            code_match = ocr['set_code'].upper() in [code.upper() for code in codes]
        language_match = None
        if candidate.get('language') and ocr.get('language') and _reliable(ocr, 'language'):
            language_match = candidate['language'].lower() == ocr['language'].lower()
        evidence = {'ocr_number': number, 'ocr_total': total, 'catalog_number': expected,
                    'catalog_total': catalog_total, 'number_match': number_match, 'total_match': total_match,
                    'ocr_name': ocr.get('name'), 'catalog_name': candidate.get('name'), 'name_match': name_match,
                    'ocr_set_code': ocr.get('set_code'), 'catalog_set_codes': codes, 'set_code_match': code_match,
                    'language_match': language_match, 'variant_verified': False, 'language_verified': language_match is True}
        score = float(candidate['similarity'])
        if not math.isfinite(score):
            raise ValueError('nonfinite_similarity')
        if candidate.get('number_is_printed') is False and not candidate.get('number'):
            candidate['number'] = None
        for match, weight in ((number_match, .2), (total_match, .1), (name_match, .15), (code_match, .15), (language_match, .15)):
            score += weight if match is True else -weight if match is False else 0
        candidate.update(evidence=evidence, fusion_score=score)
        ranked.append(candidate)
    return sorted(ranked, key=lambda candidate: candidate['fusion_score'], reverse=True)


def identity_vetoes(ranked, ocr, open_set_min_similarity, *, raw_candidates=None, require_number_match=False):
    """Pure non-tuned vetoes; live-frame eligibility additionally requires OCR number agreement."""
    reasons = []
    if not ranked or ranked[0]['similarity'] < open_set_min_similarity:
        reasons.append('low_image_similarity')
    if ranked:
        evidence = ranked[0]['evidence']
        for key, reason in (('number_match', 'ocr_number_conflict'), ('total_match', 'ocr_total_conflict'),
                            ('name_match', 'ocr_name_conflict'), ('set_code_match', 'ocr_set_code_conflict'),
                            ('language_match', 'ocr_language_conflict')):
            if evidence.get(key) is False:
                reasons.append(reason)
        visual = ranked if raw_candidates is None else raw_candidates
        strongest_visual = max(candidate['similarity'] for candidate in visual)
        if (ranked[0]['similarity'] < strongest_visual and evidence.get('number_match') is True
                and evidence.get('name_match') is not True and evidence.get('set_code_match') is not True):
            reasons.append('number_only_override')
        if len(ranked) < 2:
            reasons.append('missing_competitor')
        if require_number_match and (ocr.get('number') is None or evidence.get('number_match') is not True):
            reasons.append('ocr_missing_or_ambiguous')
    return reasons


def effective_margin(ranked):
    if len(ranked) < 2:
        return None
    score_key = 'fusion_score' if 'fusion_score' in ranked[0] else 'similarity'
    raw = ranked[0][score_key] - ranked[1].get(score_key, ranked[1]['similarity'])
    top_name = _name(ranked[0].get('name'))
    if top_name and ranked[0].get('similarity', 0) >= .85 and _name(ranked[1].get('name')) == top_name:
        competitor = next((c for c in ranked[1:] if _name(c.get('name')) != top_name), None)
        return (ranked[0][score_key] - competitor.get(score_key, competitor['similarity'])) if competitor else max(raw, .1)
    return raw


def decision(ranked, ocr, policy):
    reasons = ['single_image_requires_human_confirmation', 'variant_not_visually_verified',
               'scores_are_not_probabilities', 'uncalibrated_identity_thresholds']
    margin = ranked[0]['fusion_score'] - ranked[1]['fusion_score'] if len(ranked) > 1 else None
    distinct_margin = effective_margin(ranked)
    vetoes = identity_vetoes(ranked, ocr, policy.open_set_min_similarity)
    if ranked:
        if not ranked[0]['evidence'].get('language_verified'):
            reasons.append('language_not_visually_verified')
        if ranked[0]['similarity'] < policy.min_similarity:
            vetoes.append('weak_image_similarity')
        if distinct_margin is not None and distinct_margin < policy.min_margin:
            vetoes.append('ambiguous_margin')
    reasons.extend(vetoes)
    if ocr.get('number') is None:
        reasons.append('ocr_missing_or_ambiguous')
    return {'status': 'unknown' if vetoes else 'needs_confirmation', 'reason_codes': reasons,
            'margin': margin, 'margin_score': 'fusion_score', 'score_type': 'uncalibrated_similarity'}


def json_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_non_tuned_policy(value):
    from .recognition_config import RecognitionPolicy
    if (not isinstance(value, dict) or set(value) != set(NON_TUNED_POLICY_FIELDS)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in value.values())):
        raise ValueError('missing_or_invalid_non_tuned_policy')
    RecognitionPolicy(**value).validate()
    return {key: float(value[key]) for key in NON_TUNED_POLICY_FIELDS}


def result_fingerprints(registry, model_versions, method, *, policy):
    """Shared producer for benchmark and future consumers; policy is explicit execution evidence.

    Only per-frame non-tuned policy is bound. Temporal consensus and pose are not
    evaluated by this calibration rule; min_similarity/min_margin are its outputs.
    """
    from .recognition_config import RecognitionPolicy
    if not isinstance(policy, RecognitionPolicy):
        raise ValueError('explicit_recognition_policy_required')
    policy.validate()
    non_tuned = validate_non_tuned_policy({key: getattr(policy, key) for key in NON_TUNED_POLICY_FIELDS})
    versions = {key: value for key, value in model_versions.items()
                if key not in {'embedding', 'kind', 'gallery', 'projection', 'identity_fingerprint', 'roi'}}
    inputs = {
        'gallery': {**{key: registry.get(key) for key in
                       ('gallery_version', 'index_sha256', 'phash_sha256')},
                    'rows_sha256': json_fingerprint(registry.get('rows'))},
        'model': {**{key: registry.get(key) for key in
                     ('model', 'checkpoint_kind', 'embedding_sha256', 'projection', 'preprocessing')},
                  'method': method, 'model_versions': versions,
                  'roi': registry.get('roi') if method in ('full', 'embedding_ocr') else 'not_run',
                  'decision_rule': CALIBRATION_RULE, 'non_tuned_policy': non_tuned},
    }
    return {'gallery_fingerprint': json_fingerprint(inputs['gallery']),
            'model_fingerprint': json_fingerprint(inputs['model']), 'fingerprint_inputs': inputs}


def identity_fingerprint(registry):
    keys = ('model', 'checkpoint_kind', 'embedding_sha256', 'projection', 'preprocessing',
            'gallery_version', 'index_sha256', 'phash_sha256', 'rows')
    return hashlib.sha256(json.dumps({key: registry.get(key) for key in keys}, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()
