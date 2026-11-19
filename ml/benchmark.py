"""Labeled-image benchmark. Scores are similarities, never calibrated probabilities.

Stage rates are len(measurements) / sum(measured seconds), not webcam FPS.
Pipeline stages can include setup/postprocessing and overlap with total; missing
stages stay missing. ``top1`` remains retrieval top1, independent of rejection.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image
from cardlens_ai import pipeline
from cardlens_ai import large_gallery
from cardlens_ai.identity import json_fingerprint, result_fingerprints
from cardlens_ai.pipeline import identify, gallery, embedding, ocr_engine
from cardlens_ai.identity import CALIBRATION_RULE, NON_TUNED_POLICY_FIELDS

METHODS = ("phash", "embedding", "embedding_ocr", "full")
CARD_CORPUS_METHODS = ("embedding", "embedding_ocr", "full")
BACKENDS = ("legacy", "cardcorpus")
DECISION_STATUSES = {"needs_confirmation", "accepted", "confirmed", "identified", "matched"}


def _duration(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid timing: durations must be finite nonnegative milliseconds")
    return float(value)


def _timing_summary(values):
    values = [_duration(value) for value in values]
    total = math.fsum(values)
    if not math.isfinite(total):
        raise ValueError("Invalid timing sum")
    rate = len(values) * 1000 / total if total else None
    if rate is not None and not math.isfinite(rate):
        raise ValueError("Invalid timing rate")
    return {"sample_count": len(values), "measured_total_ms": total if values else None,
            "p50_ms": float(np.percentile(values, 50)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None,
            "effective_serial_rate_hz": rate}


def accepted_identity(result):
    """Compatibility decision: a candidate proposed for confirmation is eligible.

    This is not a claim of autonomous/human acceptance or temporal confirmation.
    Explicit reject, unknown, unrecognized statuses and empty rankings abstain.
    """
    ranked = result.get("ranked_ids") or []
    return ranked[0] if ranked and result.get("status") in DECISION_STATUSES else None


def summarize_results(results):
    """Summarize a single method/group without inventing empty or missing timings."""
    results = [{**r, "ranked_ids": r.get("ranked_ids") or []} for r in results]
    known = [r for r in results if r["ground_truth"] is not None]
    unknown = [r for r in results if r["ground_truth"] is None]
    eligible = [r for r in results if accepted_identity(r) is not None]
    correct = sum(accepted_identity(r) == r["ground_truth"] for r in eligible)
    known_eligible = [r for r in known if accepted_identity(r) is not None]
    correct_known = sum(accepted_identity(r) == r["ground_truth"] for r in known_eligible)
    wrong_known = sum(accepted_identity(r) is not None and accepted_identity(r) != r["ground_truth"]
                      for r in known)
    label = {"kind": "unknown_or_reject"}
    truth_labels = sorted({r["ground_truth"] for r in known}) + [label]
    predicted_labels = sorted({accepted_identity(r) for r in eligible}) + [label]
    counts = [[0 for _ in predicted_labels] for _ in truth_labels]
    stage_values = {"total": []}
    for result in results:
        predicted = accepted_identity(result)
        truth = result["ground_truth"] if result["ground_truth"] is not None else label
        counts[truth_labels.index(truth)][predicted_labels.index(predicted if predicted is not None else label)] += 1
        stages = result.get("stage_timing_ms", {})
        for stage, value in stages.items():
            _duration(value)
            if stage != "total":
                stage_values.setdefault(stage, []).append(value)
        stage_values["total"].append(_duration(result["latency_ms"]))
    timings = {stage: _timing_summary(values) for stage, values in sorted(stage_values.items())}
    top1 = sum(bool(r["ranked_ids"]) and r["ranked_ids"][0] == r["ground_truth"] for r in known) / len(known) if known else None
    return {
        "sample_count": len(results), "known_count": len(known), "unknown_count": len(unknown),
        "confusion_matrix": {"ground_truth_labels": truth_labels, "predicted_labels": predicted_labels,
                             "counts": counts},
        "top1": top1, "retrieval_top1": top1,
        "mrr_at_3": sum(1 / (r["ranked_ids"][:3].index(r["ground_truth"]) + 1)
                        if r["ground_truth"] in r["ranked_ids"][:3] else 0 for r in known) / len(known) if known else None,
        "top3": sum(r["ground_truth"] in r["ranked_ids"][:3] for r in known) / len(known) if known else None,
        "unknown_false_accept_rate": sum(accepted_identity(r) is not None for r in unknown) / len(unknown) if unknown else None,
        "eligible_decision_count": len(eligible),
        "decision_coverage": len(eligible) / len(results) if results else None,
        "accepted_identity_accuracy": correct / len(eligible) if eligible else None,
        "known_accepted_identity_accuracy": correct_known / len(known_eligible) if known_eligible else None,
        "correct_known_coverage": correct_known / len(known) if known else None,
        "known_wrong_accept_count": wrong_known,
        "known_wrong_accept_rate": wrong_known / len(known) if known else None,
        "latency_p50_ms": timings["total"]["p50_ms"],
        "latency_p95_ms": timings["total"]["p95_ms"], "stage_timings": timings,
    }


def _cardcorpus_versions(paired, method, roi_mode="fixed_bottom_header_v3", model_versions=None):
    roi = "not_run"
    if method in ("full", "embedding_ocr"):
        candidate_roi = (model_versions or {}).get("roi")
        roi = candidate_roi if candidate_roi not in (None, "not_run") else roi_mode
    return {**paired.model_versions, "fusion": pipeline.FUSION_VERSION,
            "pose": "external_localization", "roi": roi}


def _cardcorpus_fingerprints(paired, model_versions, method, policy, roi_mode="fixed_bottom_header_v3"):
    versions = _cardcorpus_versions(paired, method, roi_mode, model_versions)
    non_tuned = {key: getattr(policy, key) for key in NON_TUNED_POLICY_FIELDS}
    inputs = {
        "gallery": {"backend": "cardcorpus", "gallery_version": paired.model_versions["gallery"],
                    "index_release": paired.model_versions["index_release"],
                    "coverage": paired.coverage},
        "model": {"backend": "cardcorpus", "method": method, "model_versions": versions,
                  "decision_rule": CALIBRATION_RULE, "non_tuned_policy": non_tuned},
    }
    return {"gallery_fingerprint": json_fingerprint(inputs["gallery"]),
            "model_fingerprint": json_fingerprint(inputs["model"]), "fingerprint_inputs": inputs}


def evaluate(rows, output, *, backend="legacy"):
    rows = list(rows)
    output = Path(output)
    if not rows:
        raise ValueError("Empty benchmark")
    if backend not in BACKENDS:
        raise ValueError("Invalid benchmark backend")
    if output.exists():
        raise ValueError("Run already exists; choose a new run ID")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    policy = pipeline.DEFAULT_POLICY
    paired = (large_gallery.get_gallery(pipeline.DATA / "models" / "cardcorpus-scanner")
              if backend == "cardcorpus" else None)
    if paired is None:
        _, registry, _ = gallery()
        ids = {r["card_id"] for r in registry["rows"]}
        methods = METHODS
    else:
        registry = None
        ids = {row["card_id"] for row in paired.rows if row is not None}
        methods = CARD_CORPUS_METHODS
    for row in rows:
        if row.get("card_id") is not None and row["card_id"] not in ids:
            raise ValueError("Known test card absent from gallery")
        if row.get("unknown_card_id") in ids:
            raise ValueError("Unknown test card leaks into gallery")
    cold = time.perf_counter()
    with Image.open(rows[0]["path"]) as source:
        first = source.convert("RGB")
    if paired is None:
        embedding(first)
    else:
        paired.embed(first.resize((504, 704)))
    ocr_engine()
    from cardlens_ai.pipeline import read_number
    warmup_ocr = read_number(first.resize((504, 704)))
    roi_mode = (warmup_ocr.get("roi_mode") if isinstance(warmup_ocr, dict) else None) or "fixed_bottom_header_v3"
    cold_ms = _duration((time.perf_counter() - cold) * 1000)
    results = []
    for sample_index, row in enumerate(rows):
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        input_sha256 = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
        offset = sample_index % len(methods)
        for method in methods[offset:] + methods[:offset]:
            start = time.perf_counter()
            trace = identify(image, mode=row.get("mode", "cropped"), corners=row.get("corners"), method=method,
                             cardcorpus=paired)
            elapsed = _duration((time.perf_counter() - start) * 1000)
            raw_timing = {stage: _duration(value) for stage, value in trace.get("timing_ms", {}).items()}
            stages = {stage: value for stage, value in raw_timing.items() if stage != "total"}
            stages["total"] = elapsed
            # Only a combined remainder is derivable; never present it as measured OCR.
            if set(raw_timing) == {"total", "warp", "retrieval"}:
                remainder = raw_timing["total"] - raw_timing["warp"] - raw_timing["retrieval"]
                if remainder >= 0:
                    stages["post_retrieval_combined"] = remainder
            ranked = [c["card_id"] for c in (trace.get("candidates") or [])]
            versions = (_cardcorpus_versions(paired, method, roi_mode, trace["model_versions"])
                        if paired is not None else trace["model_versions"])
            fingerprints = (_cardcorpus_fingerprints(paired, versions, method, policy, roi_mode) if paired is not None
                            else result_fingerprints(registry, versions, method, policy=policy))
            result = {
                **{key: row[key] for key in ("physical_card_id", "source_group_id", "split", "calibration_role", "provenance") if key in row},
                "sample_id": row["sample_id"], "ground_truth": row.get("card_id"),
                "session_id": row["session_id"], "method": method, "ranked_ids": ranked,
                "status": trace["status"], "latency_ms": elapsed, "mode": trace["mode"],
                "ocr": {k: v for k, v in trace["ocr"].items() if k != "crop"},
                "model_versions": versions, "before_ocr": trace["before_ocr"],
                "candidates": (trace.get("candidates") or []), "corners": trace["corners"],
                "reason_codes": trace["reason_codes"], "input_sha256": input_sha256,
                "backend": backend,
                **{key: trace[key] for key in ("quality", "margin", "margin_score", "score_type", "raw_candidates") if key in trace},
                "source_kind": row.get("source_kind", "unspecified"),
                "condition": row.get("condition", "unspecified"), "timing_ms": raw_timing,
                "stage_timing_ms": stages, "probabilities_calibrated": False,
                **fingerprints,
            }
            if paired is not None:
                result["matcher_fingerprint"] = versions["identity_fingerprint"]
                result["index_fingerprint"] = versions["gallery"]
            result["accepted_identity"] = accepted_identity(result)
            results.append(result)
            print(row["sample_id"], method, ranked[0] if ranked else None, round(elapsed), flush=True)
    summary = {
        "sample_count": len(rows), "backend": backend, "gallery_count": len(ids),
        "gallery_version": paired.model_versions["gallery"] if paired is not None else registry["gallery_version"],
        "model": "cardcorpus/dinov3-small-matcher" if paired is not None else registry["model"],
        "checkpoint_kind": "external_pretrained_matcher" if paired is not None else registry["checkpoint_kind"],
        "embedding_sha256": None if paired is not None else registry.get("embedding_sha256"),
        "input_manifest_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "platform": platform.platform(), "cold_initialization_ms": cold_ms,
        "probabilities_calibrated": False,
        "limitations": ["uncalibrated unknown threshold",
                        "benchmark evaluates provided cropped/manual-corner inputs; excludes detector and temporal tracking",
                        "full equals embedding+OCR in cropped/manual modes; no temporal voting",
                        "accepted identity means a non-rejected proposal, including needs_confirmation; not automatic acceptance",
                        "stage rates are measured serial service rates, not webcam FPS; total includes overlapping stages",
                        "post_retrieval_combined is derived from rounded pipeline timings; not isolated OCR latency"],
        "methods": {}, "run_id": output.name,
        "source_kinds": sorted({r.get("source_kind", "unspecified") for r in rows}),
    }
    if paired is not None:
        summary.update(coverage=paired.coverage, model_versions=paired.model_versions,
                       matcher_fingerprint=paired.model_versions["identity_fingerprint"],
                       index_fingerprint=paired.model_versions["gallery"],
                       gallery_fingerprint=_cardcorpus_fingerprints(paired, paired.model_versions,
                                                                    "embedding", policy, roi_mode)["gallery_fingerprint"])
        summary["model_fingerprints"] = {
            method: _cardcorpus_fingerprints(paired, paired.model_versions, method, policy, roi_mode)["model_fingerprint"]
            for method in methods}
    for method in methods:
        group = [r for r in results if r["method"] == method]
        metrics = summarize_results(group)
        for field in ("source_kind", "condition"):
            metrics[f"by_{field}"] = {value: summarize_results([r for r in group if r[field] == value])
                                     for value in sorted({r[field] for r in group})}
        summary["methods"][method] = metrics
    summary_json = json.dumps(summary, indent=2, allow_nan=False)
    results_json = "\n".join(json.dumps(r, allow_nan=False) for r in results)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(summary_json, encoding="utf-8")
    (output / "results.jsonl").write_text(results_json, encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "ground_truth", "method", "ranked_ids", "status", "latency_ms"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in results)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="legacy")
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid run id")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    print(json.dumps(evaluate(rows, Path("data/benchmarks") / args.run_id, backend=args.backend), indent=2))
