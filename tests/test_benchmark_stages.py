import json

import numpy as np
import pytest
from PIL import Image

from ml import benchmark
from cardlens_ai import large_gallery, pipeline
from cardlens_ai.identity import CALIBRATION_RULE, json_fingerprint
from cardlens_ai.recognition_config import RecognitionPolicy


def test_stage_summary_uses_sum_of_measured_durations_and_handles_empty():
    rows = [{"ground_truth": "a", "ranked_ids": ["a"], "status": "needs_confirmation",
             "latency_ms": total, "stage_timing_ms": {"total": total, "retrieval": stage}}
            for total, stage in [(20., 10.), (80., 30.)]]
    metrics = benchmark.summarize_results(rows)
    assert metrics["top1"] == metrics["retrieval_top1"] == 1
    stage = metrics["stage_timings"]["retrieval"]
    assert stage == {"sample_count": 2, "measured_total_ms": 40., "p50_ms": 20.,
                     "p95_ms": 29., "effective_serial_rate_hz": 50.}
    assert metrics["stage_timings"]["total"]["effective_serial_rate_hz"] == 20
    empty = benchmark.summarize_results([])
    assert empty["latency_p50_ms"] is None
    assert empty["top1"] is None
    assert empty["stage_timings"]["total"]["effective_serial_rate_hz"] is None
    rows[0]["stage_timing_ms"]["retrieval"] = 0
    rows[1]["stage_timing_ms"]["retrieval"] = 0
    assert benchmark.summarize_results(rows)["stage_timings"]["retrieval"]["effective_serial_rate_hz"] is None


@pytest.mark.parametrize("status, ranked", [("unknown", ["a"]), ("reject", ["a"]),
                                            ("rejected", ["a"]), ("needs_confirmation", [])])
def test_unknown_reject_and_missing_candidates_are_not_identity_accepts(status, ranked):
    rows = [{"ground_truth": truth, "ranked_ids": ranked, "status": status,
             "latency_ms": 0, "stage_timing_ms": {"total": 0}} for truth in ("a", None)]
    metrics = benchmark.summarize_results(rows)
    assert metrics["top1"] == bool(ranked)
    assert metrics["decision_coverage"] == 0
    assert metrics["accepted_identity_accuracy"] is None
    assert metrics["unknown_false_accept_rate"] == 0
    assert metrics["confusion_matrix"]["predicted_labels"] == [{"kind": "unknown_or_reject"}]


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1, True])
def test_non_finite_or_invalid_timings_fail_closed(duration):
    with pytest.raises(ValueError, match="timing"):
        benchmark.summarize_results([{"ground_truth": None, "ranked_ids": [], "status": "unknown",
                                      "latency_ms": duration, "stage_timing_ms": {"total": duration}}])


def test_evaluate_preserves_stages_provenance_fingerprints_and_groups(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    registry = {"rows": [{"card_id": "set:001"}], "gallery_version": "fixture",
                "model": "fixture", "checkpoint_kind": "test_fixture", "embedding_sha256": {}}
    monkeypatch.setattr(benchmark, "gallery", lambda: (None, registry, None))
    monkeypatch.setattr(benchmark, "embedding", lambda image: None)
    monkeypatch.setattr(benchmark, "ocr_engine", lambda: None)
    monkeypatch.setattr(pipeline, "read_number", lambda image: {})
    monkeypatch.setattr(benchmark, "identify", lambda image, **kwargs: {
        "candidates": [{"card_id": "set:001"}], "before_ocr": [], "status": "reject",
        "mode": "cropped", "corners": None, "reason_codes": [], "ocr": {},
        "model_versions": {"embedding": "fixture", "gallery": "fixture"},
        "timing_ms": {"warp": 0., "retrieval": 2., "ocr": 3., "total": 5.},
    })
    rows = [{"sample_id": "one", "session_id": "session", "card_id": "set:001", "path": str(image),
             "physical_card_id": "physical", "source_group_id": "source", "split": "validation",
             "calibration_role": "fit", "source_kind": "test_fixture", "condition": "glare",
             "provenance": {"source": "fixture", "label_source": "author"}}]
    output = tmp_path / "benchmark"
    summary = benchmark.evaluate(rows, output)
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(results) == 4
    for result in results:
        assert result["physical_card_id"] == "physical"
        assert result["provenance"] == rows[0]["provenance"]
        assert result["split"] == "validation"
        assert result["condition"] == "glare"
        assert result["timing_ms"]["ocr"] == 3
        assert len(result["gallery_fingerprint"]) == len(result["model_fingerprint"]) == 64
        assert result["accepted_identity"] is None
    for method in summary["methods"].values():
        assert method["by_source_kind"]["test_fixture"]["top1"] == 1
        assert method["by_condition"]["glare"]["stage_timings"]["ocr"]["p50_ms"] == 3
        assert method["accepted_identity_accuracy"] is None
    json.dumps(summary, allow_nan=False)


def test_fingerprints_are_compact_and_cover_gallery_weights_method_and_preprocessing():
    registry = {"rows": [{"card_id": f"set:{i}", "metadata": "x" * 200} for i in range(100)],
                "gallery_version": "version", "index_sha256": "index", "phash_sha256": "phash",
                "model": "model", "embedding_sha256": {"model": "hash"}, "preprocessing": "v1"}
    first = benchmark.result_fingerprints(registry, {"fusion": "v1"}, "embedding_ocr", policy=RecognitionPolicy())
    assert len(json.dumps(first)) < 1500
    assert benchmark.result_fingerprints(registry, {"fusion": "v1"}, "embedding", policy=RecognitionPolicy())["model_fingerprint"] != first["model_fingerprint"]
    registry["embedding_sha256"] = {"model": "changed"}
    assert benchmark.result_fingerprints(registry, {"fusion": "v1"}, "embedding_ocr", policy=RecognitionPolicy())["model_fingerprint"] != first["model_fingerprint"]
    registry["rows"][0]["metadata"] = "changed"
    assert benchmark.result_fingerprints(registry, {"fusion": "v1"}, "embedding_ocr", policy=RecognitionPolicy())["gallery_fingerprint"] != first["gallery_fingerprint"]


def test_missing_rankings_abstain_and_preserve_retrieval_metrics():
    metrics = benchmark.summarize_results([{"ground_truth": "a", "status": "reject", "latency_ms": 1}])
    assert metrics["top1"] == 0
    assert metrics["top3"] == 0
    assert metrics["mrr_at_3"] == 0
    assert metrics["known_wrong_accept_count"] == 0


def test_benchmark_fingerprints_and_jsonl_feed_calibration(tmp_path, monkeypatch):
    from ml import calibrate

    registry = {"rows": [{"card_id": "set:001"}, {"card_id": "set:002"}],
                "gallery_version": "fixture", "model": "fixture", "checkpoint_kind": "test_fixture"}
    monkeypatch.setattr(benchmark, "gallery", lambda: (None, registry, None))
    monkeypatch.setattr(benchmark, "embedding", lambda image: None)
    monkeypatch.setattr(benchmark, "ocr_engine", lambda: None)
    monkeypatch.setattr(pipeline, "read_number", lambda image: {})

    def identify(image, **kwargs):
        assert kwargs["cardcorpus"] is None
        known = image.getpixel((0, 0))[0] % 2 == 0
        return {"status": "needs_confirmation", "candidates": [
            {"card_id": "set:001", "similarity": .9 if known else .6, "official_total": "100",
             "fusion_score": 1.2 if known else .9,
             "evidence": {"number_match": True, "total_match": True, "ocr_number": "001",
                          "catalog_number": "001", "ocr_total": "100", "catalog_total": "100"}},
            {"card_id": "set:002", "similarity": .4, "fusion_score": .4}],
            "mode": "cropped", "ocr": {"number": "001", "total": "100"}, "quality": {"usable": True, "sharpness": 50, "brightness": 100, "glare": 0.0},
            "model_versions": {"embedding": "fixture"}, "before_ocr": [], "corners": None,
            "reason_codes": [], "timing_ms": {"warp": 1, "retrieval": 2, "total": 4}}

    def with_raw_candidates(image, **kwargs):
        trace = identify(image, **kwargs)
        trace["raw_candidates"] = trace["candidates"]
        return trace

    monkeypatch.setattr(benchmark, "identify", with_raw_candidates)
    rows = []
    for index in range(8):
        image = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), (index, 20, 20)).save(image)
        rows.append({"sample_id": str(index), "session_id": str(index), "physical_card_id": str(index),
                     "source_group_id": str(index), "path": str(image),
                     "card_id": "set:001" if index % 2 == 0 else None,
                     "source_kind": "test_fixture", "condition": "normal", "split": "validation",
                     "calibration_role": "fit" if index < 4 else "check",
                     "provenance": {"source": f"fixture:{index}", "label_source": "fixture-author"}})
    output = tmp_path / "fixture-benchmark"
    benchmark.evaluate(rows, output)
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    selected = [result for result in results if result["method"] == "embedding_ocr"]
    manifest = {"schema_version": 1, "purpose": "threshold_calibration", "split": "validation",
                "dataset_id": "benchmark-integration-fixture",
                "gallery_fingerprint": selected[0]["gallery_fingerprint"],
                "model_fingerprint": selected[0]["model_fingerprint"],
                "provenance": {"independent_of_training": True, "independent_of_gallery": True,
                               "description": "Fixture only, not actual model measurements"},
                "samples": [{field: result[field] for field in calibrate.SAMPLE_FIELDS} for result in selected]}
    artifact = calibrate.calibrate(results, manifest, method="embedding_ocr", min_similarities=[.8],
                                   min_margins=[.2], max_unknown_far=0,
                                   max_known_wrong_accept_rate=0, min_known_coverage=1)
    assert artifact["metrics"]["check"]["accuracy_among_eligible"] == 1
    assert artifact["fingerprints"]["model"] == selected[0]["model_fingerprint"]
    assert artifact["active"] is False


def test_fingerprint_is_stable_on_quality_rejects_but_covers_projection():
    registry = {"rows": [], "model": "model", "gallery_version": "gallery",
                "embedding_sha256": {"model": "hash"}, "projection": {"sha256": "projection"},
                "preprocessing": "preprocess", "roi": "roi-version"}
    versions = {"embedding": "model", "kind": "pretrained", "gallery": "gallery",
                "fusion": "v3", "roi": "roi-version", "preprocessing": "preprocess",
                "identity_fingerprint": "active", "projection": "projection"}
    rejected = {**versions, "embedding": "not_run", "kind": "not_run", "gallery": "not_run",
                "roi": "not_run", "identity_fingerprint": "not_run", "projection": None}
    first = benchmark.result_fingerprints(registry, versions, "embedding_ocr", policy=RecognitionPolicy())
    assert benchmark.result_fingerprints(registry, rejected, "embedding_ocr", policy=RecognitionPolicy()) == first
    registry["projection"]["sha256"] = "changed"
    assert benchmark.result_fingerprints(registry, versions, "embedding_ocr", policy=RecognitionPolicy())["model_fingerprint"] != first["model_fingerprint"]


def test_known_accuracy_excludes_abstentions_and_coverage_keeps_them():
    rows = [
        {"ground_truth": "a", "ranked_ids": ["a"], "status": "needs_confirmation", "latency_ms": 1},
        {"ground_truth": "b", "ranked_ids": ["b"], "status": "unknown", "latency_ms": 1},
    ]
    metrics = benchmark.summarize_results(rows)
    assert metrics["known_accepted_identity_accuracy"] == 1
    assert metrics["correct_known_coverage"] == .5
    metrics = benchmark.summarize_results(rows[1:])
    assert metrics["known_accepted_identity_accuracy"] is None
    assert metrics["correct_known_coverage"] == 0


def test_fingerprint_binds_non_tuned_policy_but_not_selected_thresholds():
    from dataclasses import replace
    from cardlens_ai.recognition_config import RecognitionPolicy
    policy = RecognitionPolicy()
    first = benchmark.result_fingerprints({}, {}, "embedding_ocr", policy=policy)
    for field, value in (("min_brightness", 40), ("min_sharpness", 40),
                         ("max_glare", .3), ("open_set_min_similarity", .7)):
        changed = benchmark.result_fingerprints({}, {}, "embedding_ocr", policy=replace(policy, **{field: value}))
        assert changed["model_fingerprint"] != first["model_fingerprint"]
        assert changed["gallery_fingerprint"] == first["gallery_fingerprint"]
    tuned = replace(policy, min_similarity=.9, min_margin=.1)
    assert benchmark.result_fingerprints({}, {}, "embedding_ocr", policy=tuned) == first


def test_synthetic_benchmark_limitations_do_not_claim_pose_was_absent(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    registry = {"rows": [{"card_id": "set:001"}], "gallery_version": "fixture",
                "model": "fixture", "checkpoint_kind": "test_fixture"}
    monkeypatch.setattr(benchmark, "gallery", lambda: (None, registry, None))
    monkeypatch.setattr(benchmark, "embedding", lambda image: None)
    monkeypatch.setattr(benchmark, "ocr_engine", lambda: None)
    monkeypatch.setattr(pipeline, "read_number", lambda image: {})
    monkeypatch.setattr(benchmark, "identify", lambda image, **kwargs: {
        "candidates": [], "raw_candidates": [], "before_ocr": [], "status": "unknown",
        "mode": "cropped", "corners": None, "reason_codes": [], "ocr": {},
        "quality": {"usable": False}, "model_versions": {},
        "timing_ms": {"warp": 0., "retrieval": 0., "total": 1.}})
    output = tmp_path / "benchmark"
    summary = benchmark.evaluate([{"sample_id": "one", "session_id": "one", "card_id": None,
                                  "path": str(image)}], output)
    limitations = " ".join(summary["limitations"])
    assert "no pose checkpoint" not in limitations
    assert "excludes detector and temporal tracking" in limitations


@pytest.fixture
def paired_gallery(tmp_path, monkeypatch):
    vectors = np.eye(2, large_gallery.EMBED_DIM, dtype=np.float32)
    index = large_gallery.faiss.IndexFlatIP(large_gallery.EMBED_DIM)
    index.add(vectors)
    paired = large_gallery.LargeGallery(
        index, [{"card_id": "paired:001", "official_total": "100"}, None, {"card_id": "paired:002"}],
        [0, 2], lambda _: vectors[:1],
        {"embedding": "cardcorpus-fixture", "kind": "external_pretrained_matcher",
         "gallery": "a" * 64, "identity_fingerprint": "b" * 64,
         "index_release": large_gallery.RELEASE, "device": "cpu"},
        {"reference_images": 3, "mapped_reference_images": 2, "indexed_printings": 2},
    )

    def get_gallery(root, **kwargs):
        assert root == tmp_path / "models" / "cardcorpus-scanner"
        return paired

    def unexpected(*args):
        pytest.fail("Explicit CardCorpus benchmark used legacy or auto resources")

    monkeypatch.setattr(large_gallery, "get_gallery", get_gallery)
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setattr(pipeline, "active_cardcorpus_gallery", unexpected)
    for module in (benchmark, pipeline):
        monkeypatch.setattr(module, "gallery", unexpected)
        monkeypatch.setattr(module, "embedding", unexpected)
    monkeypatch.setattr(benchmark, "ocr_engine", lambda: None)
    monkeypatch.setattr(pipeline, "read_number", lambda _: {"number": "001", "total": "100", "roi_mode": "fixture"})
    return paired


def test_cardcorpus_benchmark_reports_paired_scores_and_fingerprints_on_quality_rejects(tmp_path, paired_gallery):
    sharp = tmp_path / "sharp.png"
    Image.fromarray(np.random.default_rng(31).integers(20, 230, (704, 504, 3), dtype=np.uint8)).save(sharp)
    blurred = tmp_path / "blurred.png"
    Image.new("RGB", (504, 704), "gray").save(blurred)
    rows = [{"sample_id": "known", "session_id": "fixture", "card_id": "paired:001", "path": str(sharp)},
            {"sample_id": "unknown", "session_id": "fixture", "card_id": None,
             "unknown_card_id": "unindexed:001", "path": str(blurred)}]
    output = tmp_path / "paired"
    summary = benchmark.evaluate(rows, output, backend="cardcorpus")
    assert summary["backend"] == "cardcorpus"
    assert summary["coverage"] == paired_gallery.coverage
    assert summary["gallery_count"] == 2
    assert summary["gallery_version"] == paired_gallery.model_versions["gallery"]
    assert summary["model_versions"]["identity_fingerprint"] == paired_gallery.model_versions["identity_fingerprint"]
    assert set(summary["methods"]) == {"embedding", "embedding_ocr", "full"}
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(results) == 6
    for result in results:
        assert result["backend"] == "cardcorpus"
        known = result["sample_id"] == "known"
        assert result["accepted_identity"] == ("paired:001" if known else None)
        assert result["quality"]["usable"] is known
        versions = result["fingerprint_inputs"]["model"]["model_versions"]
        assert versions["identity_fingerprint"] == paired_gallery.model_versions["identity_fingerprint"]
        assert result["fingerprint_inputs"]["model"]["decision_rule"] == CALIBRATION_RULE
        for kind in ("gallery", "model"):
            field = f"{kind}_fingerprint"
            expected = summary["gallery_fingerprint"] if kind == "gallery" else summary["model_fingerprints"][result["method"]]
            assert result[field] == expected == json_fingerprint(result["fingerprint_inputs"][kind])


@pytest.mark.parametrize("labels,message", [
    ({"card_id": "legacy:001"}, "Known test card absent from gallery"),
    ({"card_id": None, "unknown_card_id": "paired:002"}, "Unknown test card leaks into gallery"),
])
def test_cardcorpus_benchmark_checks_labels_against_mapped_rows(tmp_path, paired_gallery, labels, message):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=message):
        benchmark.evaluate([{"sample_id": "one", **labels}], output, backend="cardcorpus")
    assert not output.exists()
