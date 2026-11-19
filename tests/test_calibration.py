import copy
import hashlib
import json

import pytest

from ml import calibrate
from cardlens_ai.identity import result_fingerprints
from cardlens_ai.recognition_config import RecognitionPolicy


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def held_out():
    samples, results = [], []
    for role in ("fit", "check"):
        for kind in ("known", "unknown"):
            for index in range(2):
                sid = f"{role}-{kind}-{index}"
                sample = {
                    "sample_id": sid, "ground_truth": "set:001" if kind == "known" else None,
                    "input_sha256": digest(sid), "session_id": sid,
                    "physical_card_id": sid, "source_group_id": sid,
                    "source_kind": "test_fixture", "condition": "normal",
                    "split": "validation", "calibration_role": role,
                    "provenance": {"source": f"fixture:{sid}", "label_source": "fixture-author"},
                }
                samples.append(sample)
                number = "001"
                results.append({
                    **copy.deepcopy(sample), "method": "embedding_ocr", "status": "needs_confirmation",
                    "gallery_fingerprint": digest("gallery"), "model_fingerprint": digest("model"),
                    "reason_codes": ["single_image_requires_human_confirmation", "variant_not_visually_verified",
                                     "scores_are_not_probabilities", "uncalibrated_identity_thresholds"],
                    "ranked_ids": ["set:001", "set:002"],
                    "quality": {"usable": True, "sharpness": 50, "brightness": 100, "glare": 0.0},
                    "candidates": [
                        {"card_id": "set:001", "similarity": .9 if kind == "known" else .6,
                         "fusion_score": 1.2 if kind == "known" else .9,
                         "official_total": "100", "evidence": {
                             "number_match": True, "total_match": True,
                             "ocr_number": number, "ocr_total": "100",
                             "catalog_number": "001", "catalog_total": "100"}},
                        {"card_id": "set:002", "similarity": .5, "fusion_score": .5},
                    ],
                    "ocr": {"number": number, "total": "100"},
                })
    fingerprints = result_fingerprints({}, {}, "embedding_ocr", policy=RecognitionPolicy())
    for result in results:
        result.update(copy.deepcopy(fingerprints))
        result["raw_candidates"] = result["candidates"]
    manifest = {
        "schema_version": 1, "purpose": "threshold_calibration", "split": "validation",
        "dataset_id": "small-actual-json-fixture",
        **{key: fingerprints[key] for key in ("gallery_fingerprint", "model_fingerprint")},
        "provenance": {"description": "Authored unit fixture, no model measurements",
                       "independent_of_training": True, "independent_of_gallery": True},
        "samples": samples,
    }
    return manifest, results


def run(data, **kwargs):
    manifest, results = data
    options = dict(method="embedding_ocr", min_similarities=[.5, .8], min_margins=[0, .2],
                   max_unknown_far=0, max_known_wrong_accept_rate=0, min_known_coverage=.5)
    options.update(kwargs)
    return calibrate.calibrate(results, manifest, **options)


def test_selects_fit_threshold_and_reports_separate_check_and_raw_evidence(held_out):
    artifact = run(held_out)
    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "pkmlens_threshold_calibration"
    assert artifact["thresholds"] == {"min_similarity": .8, "min_margin": .2}
    assert artifact["probabilities_calibrated"] is False
    assert artifact["active"] is False
    assert artifact["decision_rule"]["schema"] == "live_frame_ocr_consistent_fusion_margin_v2"
    assert artifact["decision_rule"]["scope"] == "live_frame_before_temporal_consensus"
    assert artifact["decision_rule"]["single_image_proposal_requires_ocr_number"] is False
    assert artifact["decision_rule"]["temporal_consensus_evaluated"] is False
    assert len(artifact["fingerprints"]["dataset"]) == 64
    for role in ("fit", "check"):
        metrics = artifact["metrics"][role]
        assert metrics["sample_count"] == 4
        assert metrics["coverage"] == .5
        assert metrics["unknown_false_accept_rate"] == 0
        assert metrics["known_wrong_accept_count"] == 0
        assert metrics["accuracy_among_eligible"] == 1
    assert len(artifact["raw_evidence"]) == 8
    assert len(artifact["fit_search"]) == 4
    json.dumps(artifact, allow_nan=False)


@pytest.mark.parametrize("field", ["session_id", "physical_card_id", "source_group_id", "input_sha256"])
def test_rejects_leakage_and_duplicate_images(held_out, field):
    manifest, results = held_out
    shared = manifest["samples"][0][field]
    manifest["samples"][4][field] = shared
    results[4][field] = shared
    with pytest.raises(ValueError, match="overlap|duplicate"):
        run(held_out)


@pytest.mark.parametrize("mutation, message", [
    (lambda m, r: m.update(split="test"), "validation"),
    (lambda m, r: m["samples"][0].update(split="test"), "validation"),
    (lambda m, r: r[0].update(split="test"), "validation|mismatch"),
    (lambda m, r: r[0].update(gallery_fingerprint=digest("other")), "fingerprint"),
    (lambda m, r: r[0].update(model_fingerprint=digest("other")), "fingerprint"),
    (lambda m, r: m["provenance"].update(independent_of_gallery=False), "provenance"),
    (lambda m, r: r[0].pop("provenance"), "provenance|mismatch"),
    (lambda m, r: r.append(copy.deepcopy(r[0])), "duplicate"),
    (lambda m, r: r.pop(), "missing"),
    (lambda m, r: r[0]["candidates"][0].update(similarity=float("nan")), "finite"),
])
def test_rejects_invalid_evidence(held_out, mutation, message):
    mutation(*held_out)
    with pytest.raises(ValueError, match=message):
        run(held_out)


def test_requires_classes_and_independent_groups_in_both_roles(held_out):
    manifest, results = held_out
    manifest["samples"] = [s for s in manifest["samples"] if s["sample_id"] != "fit-unknown-1"]
    results[:] = [r for r in results if r["sample_id"] != "fit-unknown-1"]
    with pytest.raises(ValueError, match="insufficient"):
        run(held_out)


def test_never_retunes_on_check_partition(held_out):
    # A looser pair can fit; only the check unknowns look indistinguishable from knowns.
    for result in held_out[1]:
        if result["calibration_role"] == "check" and result["ground_truth"] is None:
            result["candidates"][0]["similarity"] = .95
    with pytest.raises(ValueError, match="calibration_check_failed"):
        run(held_out)


@pytest.mark.parametrize("change", ["number", "total", "flag", "missing_runner_up"])
def test_metadata_conflicts_and_missing_margin_are_ineligible(held_out, change):
    for result in held_out[1]:
        if result["ground_truth"] is not None:
            if change == "number":
                result["ocr"]["number"] = "002"
            elif change == "total":
                result["ocr"]["total"] = "101"
            elif change == "flag":
                result["candidates"][0]["evidence"]["number_match"] = False
            else:
                result["candidates"] = result["candidates"][:1]
                result["ranked_ids"] = result["ranked_ids"][:1]
    with pytest.raises(ValueError, match="no_feasible_thresholds"):
        run(held_out)


def test_no_forced_threshold_for_empty_or_unusable_search(held_out):
    with pytest.raises(ValueError, match="empty|insufficient"):
        calibrate.calibrate([], held_out[0], method="embedding_ocr", min_similarities=[.8],
                            min_margins=[.2], max_unknown_far=0,
                            max_known_wrong_accept_rate=0, min_known_coverage=.5)
    with pytest.raises(ValueError, match="no_feasible_thresholds"):
        run(held_out, min_similarities=[1])
    with pytest.raises(ValueError, match="method"):
        run(held_out, method="unreviewed")
    with pytest.raises(ValueError, match="bounded|grid"):
        run(held_out, min_similarities=[.8] * 1000)


def test_cli_reads_jsonl_and_refuses_failed_or_existing_output(held_out, tmp_path):
    manifest, results = held_out
    mpath, rpath, output = tmp_path / "manifest.json", tmp_path / "results.jsonl", tmp_path / "artifact.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    rpath.write_text("\n".join(json.dumps(row) for row in results), encoding="utf-8")
    argv = [str(rpath), "--manifest", str(mpath), "--method", "embedding_ocr", "--output", str(output),
            "--min-similarities", "0.5,0.8", "--min-margins", "0,0.2", "--max-unknown-far", "0",
            "--max-known-wrong-accept-rate", "0", "--min-known-coverage", "0.5"]
    assert calibrate.main(argv) == 0
    assert json.loads(output.read_text())["thresholds"]["min_similarity"] == .8
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        calibrate.main(argv)
    assert output.read_bytes() == original
    output.unlink()
    results[0]["split"] = "test"
    rpath.write_text("\n".join(json.dumps(row) for row in results), encoding="utf-8")
    with pytest.raises(SystemExit):
        calibrate.main(argv)
    assert not output.exists()


def test_linked_capture_groups_do_not_count_as_independent(held_out):
    manifest, results = held_out
    # Each individual field has >=2 values, but all fit observations share a capture chain.
    for sample in manifest["samples"][:4]:
        sample["source_group_id"] = "left" if sample["sample_id"].endswith("0") else "right"
        sample["session_id"] = "session-" + sample["sample_id"][-1]
        if sample["ground_truth"] is None:
            sample["source_group_id"] = "right" if sample["sample_id"].endswith("0") else "left"
    for row, sample in zip(results, manifest["samples"]):
        row.update({key: sample[key] for key in ("source_group_id", "session_id")})
    with pytest.raises(ValueError, match="insufficient"):
        run(held_out)


def test_unknown_accepts_and_known_wrong_accepts_use_correct_denominators(held_out):
    manifest, results = held_out
    for sample in manifest["samples"]:
        if sample["sample_id"].endswith("known-1") and "unknown" not in sample["sample_id"]:
            sample["ground_truth"] = "set:003"
    for row, sample in zip(results, manifest["samples"]):
        row["ground_truth"] = sample["ground_truth"]
        if row["sample_id"].endswith("unknown-1"):
            row["candidates"][0]["similarity"] = .9
    artifact = run(held_out, max_unknown_far=.5, max_known_wrong_accept_rate=.5)
    for metrics in artifact["metrics"].values():
        assert metrics["unknown_false_accept_count"] == metrics["known_wrong_accept_count"] == 1
        assert metrics["unknown_false_accept_rate"] == metrics["known_wrong_accept_rate"] == .5
        assert metrics["coverage"] == .75
        assert metrics["accuracy_among_eligible"] == 1 / 3
        assert metrics["correct_known_coverage"] == .5


def test_rejects_contradictory_training_or_gallery_provenance(held_out):
    manifest, results = held_out
    for row in (manifest["samples"][0], results[0]):
        row["source_kind"] = "gallery_reference"
    with pytest.raises(ValueError, match="provenance|independent"):
        run(held_out)


def test_rejects_inconsistent_labels_for_one_physical_card(held_out):
    manifest, results = held_out
    for row in (manifest["samples"][2], results[2]):
        row["physical_card_id"] = manifest["samples"][0]["physical_card_id"]
    with pytest.raises(ValueError, match="physical.*label"):
        run(held_out)


def test_fingerprint_payload_must_agree_with_digest(held_out):
    held_out[1][0]["fingerprint_inputs"] = {"gallery": {"version": "other"}, "model": {}}
    with pytest.raises(ValueError, match="fingerprint"):
        run(held_out)


def test_selected_methods_filter_without_reusing_other_methods(held_out):
    manifest, results = held_out
    alternate = copy.deepcopy(results)
    for row in alternate:
        row["method"] = "phash"
        row["model_fingerprint"] = digest("phash")
    results.extend(alternate)
    artifact = run(held_out)
    assert len(artifact["raw_evidence"]) == 8


@pytest.mark.parametrize("status", ["reject", "rejected", "stabilizing"])
def test_rejected_or_unconfirmed_observations_never_become_threshold_accepts(held_out, status):
    for result in held_out[1]:
        result["status"] = status
    with pytest.raises(ValueError, match="no_feasible_thresholds"):
        run(held_out)


@pytest.mark.parametrize("key,value", [("split", "train"), ("independent_of_training", False),
                                      ("independent_of_gallery", False)])
def test_sample_provenance_cannot_contradict_validation_manifest(held_out, key, value):
    for row in (held_out[0]["samples"][0], held_out[1][0]):
        row["provenance"][key] = value
    with pytest.raises(ValueError, match="provenance"):
        run(held_out)


def test_identity_with_language_and_explicit_catalog_number_is_supported(held_out):
    manifest, results = held_out
    for sample in manifest["samples"]:
        if sample["ground_truth"]:
            sample["ground_truth"] = "set:001:en"
    for result, sample in zip(results, manifest["samples"]):
        result["ground_truth"] = sample["ground_truth"]
        result["ranked_ids"] = ["set:001:en", "set:002:en"]
        for candidate, card_id in zip(result["candidates"], result["ranked_ids"]):
            candidate["card_id"] = card_id
            candidate["number"] = card_id.split(":")[1]
    artifact = run(held_out)
    assert artifact["metrics"]["check"]["correct_known_coverage"] == 1


@pytest.mark.parametrize("conflict", ["name_match", "set_code_match", "language_match"])
def test_any_ocr_metadata_contradiction_blocks_identity(held_out, conflict):
    for result in held_out[1]:
        if result["ground_truth"]:
            result["candidates"][0]["evidence"][conflict] = False
    with pytest.raises(ValueError, match="no_feasible_thresholds"):
        run(held_out)


def test_margin_uses_fusion_score_and_not_raw_similarity_difference(held_out):
    for result in held_out[1]:
        result["candidates"][0]["fusion_score"] = 1.0
        result["candidates"][1]["fusion_score"] = .99
    with pytest.raises(ValueError, match="no_feasible_thresholds"):
        run(held_out, min_margins=[.2])


def test_old_threshold_rejection_cannot_hide_false_accepts(held_out):
    for result in held_out[1]:
        if result["ground_truth"] is None:
            result["status"] = "unknown"
            result["reason_codes"] = ["weak_image_similarity"]
            result["candidates"][0]["similarity"] = .95
    with pytest.raises(ValueError, match="no_feasible_thresholds"):
        run(held_out)


def test_missing_fusion_margin_evidence_fails_closed(held_out):
    held_out[1][0]["candidates"][0].pop("fusion_score", None)
    with pytest.raises(ValueError, match="fusion_score"):
        run(held_out)


def test_missing_quality_evidence_does_not_assert_quality_passed(held_out):
    held_out[1][0].pop("quality", None)
    with pytest.raises(ValueError, match="quality evidence"):
        run(held_out)


def test_runtime_number_only_override_never_passes_calibration(held_out):
    from cardlens_ai.identity import decision, rank_candidates
    from cardlens_ai.recognition_config import RecognitionPolicy
    policy = RecognitionPolicy()
    for row in held_out[1]:
        if row["ground_truth"] is None:
            continue
        ranked = rank_candidates([
            {"card_id": "set:001", "similarity": .90, "number": "001", "official_total": "100"},
            {"card_id": "set:002", "similarity": .95, "number": "002", "official_total": "100"},
        ], row["ocr"])
        row.update(decision(ranked, row["ocr"], policy))
        assert "number_only_override" in row["reason_codes"]
        row["candidates"] = row["raw_candidates"] = ranked
        row["ranked_ids"] = [candidate["card_id"] for candidate in ranked]
    with pytest.raises(ValueError, match="no_feasible_thresholds"):
        run(held_out)


def test_calibration_cannot_retune_the_open_set_floor(held_out):
    from ml.benchmark import result_fingerprints
    from cardlens_ai.recognition_config import RecognitionPolicy
    policy = RecognitionPolicy(min_similarity=.5, open_set_min_similarity=.95)
    fingerprints = result_fingerprints({}, {}, "embedding_ocr", policy=policy)
    held_out[0].update({key: fingerprints[key] for key in ("gallery_fingerprint", "model_fingerprint")})
    for row in held_out[1]:
        row.update(copy.deepcopy(fingerprints))
    with pytest.raises(ValueError, match="no_feasible_thresholds"):
        run(held_out)


def test_legacy_policy_free_results_cannot_produce_a_calibration_artifact(held_out):
    manifest, results = held_out
    from cardlens_ai.identity import json_fingerprint
    old_gallery = {"version": "old"}
    old_model = {"method": "embedding_ocr", "fusion": "ocr-identity-v3"}
    for row in results:
        row["fingerprint_inputs"] = {"gallery": old_gallery, "model": old_model}
        row["gallery_fingerprint"] = json_fingerprint(old_gallery)
        row["model_fingerprint"] = json_fingerprint(old_model)
    manifest["gallery_fingerprint"] = results[0]["gallery_fingerprint"]
    manifest["model_fingerprint"] = results[0]["model_fingerprint"]
    with pytest.raises(ValueError, match="decision rule|policy"):
        run(held_out)


def test_quality_measurements_must_agree_with_fingerprinted_policy(held_out):
    for row in held_out[1]:
        row["quality"]["brightness"] = 10
    with pytest.raises(ValueError, match="quality evidence contradicts"):
        run(held_out)


def test_policy_change_breaks_cross_result_fingerprint_contract(held_out):
    from cardlens_ai.identity import json_fingerprint
    result = held_out[1][0]
    changed_inputs = copy.deepcopy(result["fingerprint_inputs"])
    changed_inputs["model"]["non_tuned_policy"]["open_set_min_similarity"] = .99
    result["fingerprint_inputs"] = changed_inputs
    result["model_fingerprint"] = json_fingerprint(changed_inputs["model"])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run(held_out)


def test_missing_reason_code_evidence_fails_closed(held_out):
    held_out[1][0].pop("reason_codes")
    with pytest.raises(ValueError, match="reason code evidence"):
        run(held_out)
