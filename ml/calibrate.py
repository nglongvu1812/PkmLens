"""Offline, bounded threshold selection on explicitly held-out validation results.

Usage (module mode, no models are loaded)::

    python -m ml.calibrate results.jsonl --manifest validation.json \
      --method embedding_ocr --output thresholds.json \
      --min-similarities 0.75,0.8,0.85 --min-margins 0.02,0.04 \
      --max-unknown-far 0 --max-known-wrong-accept-rate 0 \
      --min-known-coverage 0.5

Manifest v1 is a JSON object with purpose="threshold_calibration", split="validation",
dataset_id, gallery_fingerprint, model_fingerprint, provenance, and samples. Global
provenance must assert independent_of_training=true and independent_of_gallery=true
and describe its source. Each sample and result must agree on sample_id,
ground_truth (null for unknown), input_sha256, session_id, physical_card_id,
source_group_id, source_kind, condition, split="validation", calibration_role
("fit" or "check"), and provenance={source, label_source}. These are provenance
attestations, not automatic proof that a human labeled data independently. Do not
use test, training, gallery crops, or renamed copies of them as calibration data.

Artifact v1 has artifact_type="pkmlens_threshold_calibration", method, thresholds
{min_similarity,min_margin}, fingerprints {dataset,gallery,model,results}, metrics
{fit,check}, counts, constraints, fit_search and raw_evidence. Fingerprints must
match runtime inputs before any future optional consumption. The acceptance rule
requires two candidates, finite raw similarities and fusion-score margin, OCR
number agreement and no contradictory number/total/name/set-code/language evidence.
Only similarity/margin threshold unknowns are recomputed; explicit rejects,
non-threshold vetoes and quality failures remain ineligible. Measured quality.usable,
raw candidates and explicit fingerprint-bound non-tuned policy evidence are required.
This is live-frame eligibility before temporal consensus; single-image proposals
may omit OCR number evidence and are not the calibrated decision rule.
It does not turn similarity/fusion scores into probabilities,
measure temporal consensus, or establish webcam performance. No runtime policy,
registry, model or existing artifact is changed. A failed independent check never
retunes thresholds or writes an artifact.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from cardlens_ai.identity import (CALIBRATION_RULE, identity_vetoes, json_fingerprint as _hash,
                                 validate_non_tuned_policy)

ALLOWED_METHODS = ("phash", "embedding", "embedding_ocr", "full")
SCHEMA_VERSION = 1
MAX_SAMPLES = 20_000
MAX_GRID_POINTS = 1024
MAX_GRID_VALUES = 64
DECISION_STATUSES = {"needs_confirmation", "accepted", "confirmed", "identified", "matched"}
SAMPLE_FIELDS = ("sample_id", "ground_truth", "input_sha256", "session_id", "physical_card_id",
                 "source_group_id", "source_kind", "condition", "split", "calibration_role", "provenance")
GROUP_FIELDS = ("session_id", "physical_card_id", "source_group_id")


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _fingerprint(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric evidence")
    return float(value)


def _grid(values, label):
    if not values or len(values) > MAX_GRID_VALUES:
        raise ValueError(f"bounded grid requires 1..{MAX_GRID_VALUES} {label} values")
    converted = [_finite(value, label) for value in values]
    if any(not 0 <= value <= 1 for value in converted):
        raise ValueError(f"{label} grid outside supported [0,1] policy range")
    return sorted(set(converted))


def _independent_groups(samples):
    """Count connected components, not distinct IDs linked by other group keys."""
    parents = list(range(len(samples)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, sample in enumerate(samples):
        for field in GROUP_FIELDS:
            key = (field, sample[field])
            if key in owners:
                parents[root(index)] = root(owners[key])
            else:
                owners[key] = index
    return [root(index) for index in range(len(samples))]


def _validate_inputs(results, manifest, method):
    if method not in ALLOWED_METHODS:
        raise ValueError("method must be explicitly allowed")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema_version")
    if manifest.get("split") != "validation" or manifest.get("purpose") != "threshold_calibration":
        raise ValueError("validation-only threshold_calibration manifest required; no test optimization")
    if not _nonempty(manifest.get("dataset_id")):
        raise ValueError("dataset_id required")
    provenance = manifest.get("provenance", {})
    if (not isinstance(provenance, dict) or provenance.get("independent_of_training") is not True
            or provenance.get("independent_of_gallery") is not True or not _nonempty(provenance.get("description"))):
        raise ValueError("independent validation provenance required")
    for field in ("gallery_fingerprint", "model_fingerprint"):
        if not _fingerprint(manifest.get(field)):
            raise ValueError(f"manifest {field} must be SHA-256")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("empty or insufficient/beyond bounded manifest samples")
    if not isinstance(results, list) or not results:
        raise ValueError("empty results: insufficient calibration evidence")
    if len(results) > MAX_SAMPLES * len(ALLOWED_METHODS):
        raise ValueError("results exceed bounded sample limit")
    expected, images, physical_labels = {}, set(), {}
    for sample in samples:
        if not isinstance(sample, dict) or any(field not in sample for field in SAMPLE_FIELDS):
            raise ValueError("missing sample metadata/provenance")
        if sample["split"] != "validation":
            raise ValueError("every calibration sample must be validation-only")
        if sample["calibration_role"] not in ("fit", "check"):
            raise ValueError("explicit fit/check calibration_role required")
        for field in ("sample_id", *GROUP_FIELDS, "source_kind", "condition"):
            if not _nonempty(sample[field]) or sample[field] == "unspecified":
                raise ValueError(f"missing {field} provenance")
        if sample["ground_truth"] is not None and not _nonempty(sample["ground_truth"]):
            raise ValueError("ground_truth must be known card ID or null")
        if not _fingerprint(sample["input_sha256"]):
            raise ValueError("input_sha256 fingerprint required")
        if set(re.split(r"[_ -]", sample["source_kind"].lower())) & {"gallery", "reference", "train", "training"}:
            raise ValueError("source_kind contradicts independent validation provenance")
        physical = sample["physical_card_id"]
        if physical in physical_labels and physical_labels[physical] != sample["ground_truth"]:
            raise ValueError("physical card has inconsistent ground-truth labels")
        physical_labels[physical] = sample["ground_truth"]
        source = sample["provenance"]
        if not isinstance(source, dict) or not all(_nonempty(source.get(key)) for key in ("source", "label_source")):
            raise ValueError("sample source and label_source provenance required")
        if (source.get("split", "validation") != "validation"
                or any(source.get(key, True) is not True for key in ("independent_of_training", "independent_of_gallery"))):
            raise ValueError("sample provenance contradicts validation independence")
        if sample["sample_id"] in expected or sample["input_sha256"] in images:
            raise ValueError("duplicate sample ID or input image")
        expected[sample["sample_id"]] = sample
        images.add(sample["input_sha256"])
    selected, seen = [], set()
    for result in results:
        if not isinstance(result, dict) or result.get("method") not in ALLOWED_METHODS:
            raise ValueError("result method not allowed")
        key = (result.get("sample_id"), result["method"])
        if key in seen:
            raise ValueError("duplicate sample/method result")
        seen.add(key)
        sample = expected.get(result.get("sample_id"))
        if sample is None:
            raise ValueError("result sample missing from manifest")
        if any(field not in result or result[field] != sample[field] for field in SAMPLE_FIELDS):
            raise ValueError("result/manifest metadata or provenance mismatch; validation-only data required")
        if result["method"] != method:
            continue
        for field in ("gallery_fingerprint", "model_fingerprint"):
            if result.get(field) != manifest[field]:
                raise ValueError(f"result {field} mismatch")
        inputs = result.get("fingerprint_inputs")
        if (not isinstance(inputs, dict) or set(inputs) != {"gallery", "model"}
                or any(_hash(inputs[kind]) != result[f"{kind}_fingerprint"] for kind in ("gallery", "model"))):
            raise ValueError("missing or mismatched fingerprint inputs/policy evidence")
        model = inputs["model"]
        if (not isinstance(model, dict) or model.get("method") != method
                or model.get("decision_rule") != CALIBRATION_RULE):
            raise ValueError("unsupported fingerprint decision rule or method")
        validate_non_tuned_policy(model.get("non_tuned_policy"))
        if "ground_truth" not in result or not _nonempty(result.get("status")):
            raise ValueError("missing labeled result/status")
        selected.append(result)
    if {r["sample_id"] for r in selected} != set(expected):
        raise ValueError("missing selected-method results for manifest samples")
    for field in GROUP_FIELDS:
        fit = {s[field] for s in samples if s["calibration_role"] == "fit"}
        check = {s[field] for s in samples if s["calibration_role"] == "check"}
        if fit & check:
            raise ValueError(f"fit/check {field} overlap")
    components = _independent_groups(samples)
    for role in ("fit", "check"):
        for known in (True, False):
            independent = {component for sample, component in zip(samples, components)
                           if sample["calibration_role"] == role and (sample["ground_truth"] is not None) == known}
            # Two independent groups per class is only a feasibility floor, not statistical assurance.
            if len(independent) < 2:
                raise ValueError(f"insufficient independent known/unknown groups in {role}")
    dataset_hash = _hash({k: v for k, v in manifest.items() if k != "dataset_fingerprint"})
    if manifest.get("dataset_fingerprint", dataset_hash) != dataset_hash:
        raise ValueError("dataset fingerprint mismatch")
    return selected, dataset_hash


def _evidence(result):
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("missing candidate evidence")
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not _nonempty(candidate.get("card_id")):
            raise ValueError("invalid candidate identity")
        _finite(candidate.get("similarity"), "similarity")
        _finite(candidate.get("fusion_score"), "fusion_score")
        ids.append(candidate["card_id"])
    if len(set(ids)) != len(ids) or result.get("ranked_ids") != ids:
        raise ValueError("candidate ranking/identity mismatch")
    raw = result.get("raw_candidates")
    if not isinstance(raw, list):
        raise ValueError("missing raw candidate evidence for non-threshold vetoes")
    raw_scores = {}
    for candidate in raw:
        if (not isinstance(candidate, dict) or not _nonempty(candidate.get("card_id"))
                or candidate["card_id"] in raw_scores):
            raise ValueError("invalid raw candidate identity")
        raw_scores[candidate["card_id"]] = _finite(candidate.get("similarity"), "raw similarity")
    if any(raw_scores.get(c["card_id"]) != c["similarity"] for c in candidates):
        raise ValueError("raw/ranked candidate evidence mismatch")
    best = candidates[0] if candidates else {}
    score = best.get("similarity")
    margin = best["fusion_score"] - candidates[1]["fusion_score"] if len(candidates) >= 2 else None
    if margin is not None:
        _finite(margin, "margin")
    ocr = result.get("ocr")
    if not isinstance(ocr, dict):
        raise ValueError("missing OCR evidence")
    evidence = best.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("invalid candidate OCR evidence")
    number, total = ocr.get("number"), ocr.get("total")
    parts = best.get("card_id", "").split(":")
    expected = best.get("number") or (parts[1] if len(parts) > 1 else None)
    number_ok = (isinstance(number, str) and isinstance(expected, str) and bool(expected) and number.upper() == expected.upper()
                 and evidence.get("number_match") is True
                 and evidence.get("ocr_number") == number and evidence.get("catalog_number") == expected)
    catalog_total = best.get("official_total")
    total_ok = (evidence.get("total_match") is not False and evidence.get("ocr_total") == total
                and evidence.get("catalog_total") == catalog_total)
    if total is not None:
        total_ok = total_ok and isinstance(total, str) and (catalog_total is None or total == catalog_total)
    if total is not None and catalog_total is not None:
        total_ok = total_ok and evidence.get("total_match") is True
    match_keys = ("number_match", "total_match", "name_match", "set_code_match", "language_match")
    if any(evidence.get(key) is not None and not isinstance(evidence[key], bool) for key in match_keys):
        raise ValueError("invalid OCR metadata consistency flags")
    metadata_ok = not any(evidence.get(key) is False for key in match_keys)
    quality = result.get("quality")
    if not isinstance(quality, dict) or not isinstance(quality.get("usable"), bool):
        raise ValueError("missing or invalid quality evidence")
    policy = validate_non_tuned_policy(result["fingerprint_inputs"]["model"].get("non_tuned_policy"))
    sharpness = _finite(quality.get("sharpness"), "quality sharpness")
    brightness = _finite(quality.get("brightness"), "quality brightness")
    glare = _finite(quality.get("glare"), "quality glare")
    quality_meets_policy = (sharpness + .005 >= policy["min_sharpness"]
                            and brightness >= policy["min_brightness"]
                            and glare <= policy["max_glare"] + .0005)
    quality_ok = quality["usable"] and quality_meets_policy
    if quality["usable"] and not quality_meets_policy:
        raise ValueError("quality evidence contradicts fingerprinted policy")
    vetoes = identity_vetoes(candidates, ocr, policy["open_set_min_similarity"],
                             raw_candidates=raw, require_number_match=True)
    reasons = result.get("reason_codes")
    if not isinstance(reasons, list) or any(not _nonempty(reason) for reason in reasons):
        raise ValueError("invalid reason code evidence")
    # Only these two gates are retuned; all other original vetoes survive.
    retuned = {"weak_image_similarity", "ambiguous_margin"}
    informational = {"single_image_requires_human_confirmation", "variant_not_visually_verified",
                     "scores_are_not_probabilities", "uncalibrated_identity_thresholds",
                     "language_not_visually_verified"}
    preserved = [reason for reason in reasons if reason not in retuned | informational]
    proposal = (result["status"] in DECISION_STATUSES
                or result["status"] == "unknown" and bool(set(reasons) & retuned))
    prerequisites = (proposal and quality_ok and not vetoes and not preserved
                     and number_ok and total_ok and metadata_ok)
    return {**{key: result[key] for key in SAMPLE_FIELDS},
            "original_status": result["status"], "candidate_id": best.get("card_id"),
            "similarity": score, "margin": margin, "ocr_number_consistent": number_ok,
            "ocr_total_consistent": total_ok, "ocr_metadata_consistent": metadata_ok,
            "quality_usable": quality_ok, "prerequisites_met": bool(prerequisites),
            "nonthreshold_vetoes": sorted(set(vetoes + preserved)),
            "candidates": candidates, "raw_candidates": raw, "ocr": ocr,
            "reason_codes": result.get("reason_codes", []), "quality": quality}


def _decisions(rows, min_similarity, min_margin):
    return [bool(row["prerequisites_met"] and row["similarity"] >= min_similarity
                 and row["margin"] >= min_margin) for row in rows]


def _metrics(rows, decisions):
    known = sum(row["ground_truth"] is not None for row in rows)
    unknown = len(rows) - known
    eligible = sum(decisions)
    correct = sum(accept and row["candidate_id"] == row["ground_truth"] for row, accept in zip(rows, decisions))
    false_accept = sum(accept and row["ground_truth"] is None for row, accept in zip(rows, decisions))
    wrong_known = sum(accept and row["ground_truth"] is not None and row["candidate_id"] != row["ground_truth"]
                      for row, accept in zip(rows, decisions))
    known_accept = eligible - false_accept
    return {"sample_count": len(rows), "known_count": known, "unknown_count": unknown,
            "eligible_decision_count": eligible, "known_eligible_count": known_accept,
            "correct_identity_count": correct, "unknown_false_accept_count": false_accept,
            "known_wrong_accept_count": wrong_known,
            "coverage": eligible / len(rows) if rows else None,
            "known_coverage": known_accept / known if known else None,
            "correct_known_coverage": correct / known if known else None,
            "unknown_false_accept_rate": false_accept / unknown if unknown else None,
            "known_wrong_accept_rate": wrong_known / known if known else None,
            "accuracy_among_eligible": correct / eligible if eligible else None,
            "ocr_number_consistent_count": sum(row["ocr_number_consistent"] for row in rows),
            "ocr_total_consistent_count": sum(row["ocr_total_consistent"] for row in rows)}


def _feasible(metrics, constraints):
    return (metrics["correct_identity_count"] > 0
            and metrics["unknown_false_accept_rate"] <= constraints["max_unknown_far"]
            and metrics["known_wrong_accept_rate"] <= constraints["max_known_wrong_accept_rate"]
            and metrics["correct_known_coverage"] >= constraints["min_known_coverage"])


def calibrate(results, manifest, *, method, min_similarities, min_margins,
              max_unknown_far, max_known_wrong_accept_rate, min_known_coverage):
    """Return an inactive review artifact, or raise without forcing a threshold.

    Search is explicit and bounded. Optimize correct known coverage on fit only,
    then fewer bad accepts, then stricter thresholds. Check cannot choose a pair.
    min_known_coverage constrains *correct* known decisions, not wrong accepts.
    """
    similarities = _grid(min_similarities, "min_similarity")
    margins = _grid(min_margins, "min_margin")
    if len(similarities) * len(margins) > MAX_GRID_POINTS:
        raise ValueError("bounded grid exceeds maximum threshold pairs")
    constraints = {"max_unknown_far": _finite(max_unknown_far, "max_unknown_far"),
                   "max_known_wrong_accept_rate": _finite(max_known_wrong_accept_rate, "max_known_wrong_accept_rate"),
                   "min_known_coverage": _finite(min_known_coverage, "min_known_coverage")}
    if (any(not 0 <= value <= 1 for value in constraints.values())
            or constraints["min_known_coverage"] <= 0):
        raise ValueError("constraints must be [0,1] with positive min_known_coverage")
    selected, dataset_hash = _validate_inputs(results, manifest, method)
    evidence = [_evidence(result) for result in selected]
    fit = [row for row in evidence if row["calibration_role"] == "fit"]
    check = [row for row in evidence if row["calibration_role"] == "check"]
    search, feasible = [], []
    for similarity in similarities:
        for margin in margins:
            metrics = _metrics(fit, _decisions(fit, similarity, margin))
            candidate = {"min_similarity": similarity, "min_margin": margin,
                         "metrics": metrics, "feasible": _feasible(metrics, constraints)}
            search.append(candidate)
            if candidate["feasible"]:
                feasible.append(candidate)
    if not feasible:
        raise ValueError("no_feasible_thresholds: insufficient fit evidence for requested constraints")
    best = max(feasible, key=lambda c: (c["metrics"]["correct_identity_count"],
                                      -c["metrics"]["unknown_false_accept_count"],
                                      -c["metrics"]["known_wrong_accept_count"],
                                      c["min_similarity"], c["min_margin"]))
    thresholds = {"min_similarity": best["min_similarity"], "min_margin": best["min_margin"]}
    check_metrics = _metrics(check, _decisions(check, **thresholds))
    if not _feasible(check_metrics, constraints):
        raise ValueError("calibration_check_failed: frozen fit thresholds fail independent check; no artifact")
    decisions = _decisions(evidence, **thresholds)
    raw = [{**row, "eligible": accept, "accepted_identity": row["candidate_id"] if accept else None}
           for row, accept in zip(evidence, decisions)]
    counts = {role: {"samples": len(rows), "independent_groups": len(set(_independent_groups(rows))),
                     **{field: len({row[field] for row in rows}) for field in GROUP_FIELDS}}
              for role, rows in (("fit", fit), ("check", check))}
    return {
        "schema_version": SCHEMA_VERSION, "artifact_type": "pkmlens_threshold_calibration",
        "method": method, "active": False, "probabilities_calibrated": False,
        "dataset_id": manifest["dataset_id"],
        "fingerprints": {"dataset": dataset_hash, "gallery": manifest["gallery_fingerprint"],
                         "model": manifest["model_fingerprint"], "results": _hash(selected)},
        "thresholds": thresholds, "constraints": constraints, "counts": counts,
        "non_tuned_policy": selected[0]["fingerprint_inputs"]["model"]["non_tuned_policy"],
        "metrics": {"fit": best["metrics"], "check": check_metrics},
        "fit_search": search, "raw_evidence": raw,
        "validation": {"split": "validation", "strategy": "explicit_group_disjoint_fit_check",
                       "group_fields": list(GROUP_FIELDS), "minimum_groups_per_class_per_partition": 2,
                       "check_passed": True, "thresholds_selected_on": "fit_only"},
        "decision_rule": {"schema": CALIBRATION_RULE,
                           "scope": "live_frame_before_temporal_consensus",
                           "single_image_proposal_requires_ocr_number": False,
                          "score": "candidates[0].similarity", "comparison": ">=",
                          "margin": "candidates[0].fusion_score - candidates[1].fusion_score",
                          "requires": ["not_explicit_reject", "quality_usable", "two_candidates", "ocr_number_match",
                                       "no_ocr_metadata_contradiction"],
                          "old_threshold_unknown_status": "recomputed_from_evidence",
                          "temporal_consensus_evaluated": False},
        "provenance": manifest["provenance"],
        "limitations": ["Threshold calibration is not probability calibration; scores are not probabilities.",
                        "Empirical validation rates have no population FAR guarantee; two groups is only a feasibility floor.",
                        "Provenance independence is supplied by the manifest author, not automatically verified.",
                        "No test-split optimization, physical-webcam evaluation, or temporal consensus evaluation.",
                        "Artifact is inactive; runtime must validate schema, method, decision rule and all matching fingerprints."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", choices=ALLOWED_METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-similarities", required=True, help="explicit comma-separated [0,1] grid")
    parser.add_argument("--min-margins", required=True, help="explicit comma-separated [0,1] grid")
    parser.add_argument("--max-unknown-far", type=float, required=True)
    parser.add_argument("--max-known-wrong-accept-rate", type=float, required=True)
    parser.add_argument("--min-known-coverage", type=float, required=True, help="minimum correct known coverage")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output exists; choose a new artifact path")
        manifest_bytes = args.manifest.read_bytes()
        result_bytes = args.results.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        results = [json.loads(line) for line in result_bytes.decode("utf-8-sig").splitlines() if line.strip()]
        artifact = calibrate(results, manifest, method=args.method,
                             min_similarities=[float(v) for v in args.min_similarities.split(",")],
                             min_margins=[float(v) for v in args.min_margins.split(",")],
                             max_unknown_far=args.max_unknown_far,
                             max_known_wrong_accept_rate=args.max_known_wrong_accept_rate,
                             min_known_coverage=args.min_known_coverage)
        artifact["input_files_sha256"] = {"manifest": hashlib.sha256(manifest_bytes).hexdigest(),
                                          "results": hashlib.sha256(result_bytes).hexdigest()}
        encoded = json.dumps(artifact, indent=2, allow_nan=False)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except (ValueError, OSError, TypeError, KeyError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output), "thresholds": artifact["thresholds"],
                      "active": False, "probabilities_calibrated": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
