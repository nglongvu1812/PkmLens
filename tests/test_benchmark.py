import json

import pytest
from PIL import Image

from ml import benchmark
from cardlens_ai import pipeline


@pytest.fixture
def benchmark_run(tmp_path, monkeypatch):
    image = tmp_path / "card.png"
    Image.new("RGB", (10, 10)).save(image)
    registry = {
        "rows": [{"card_id": "a"}, {"card_id": "b"}],
        "gallery_version": "test", "model": "stub",
        "checkpoint_kind": "test_fixture",
    }
    monkeypatch.setattr(benchmark, "gallery", lambda: (None, registry, None))
    monkeypatch.setattr(benchmark, "embedding", lambda image: None)
    monkeypatch.setattr(benchmark, "ocr_engine", lambda: None)
    monkeypatch.setattr(pipeline, "read_number", lambda image: {})

    def run(truths, predictions):
        rows = [{"sample_id": str(i), "session_id": "fixture", "card_id": truth,
                 "path": str(image), "source_kind": "test_fixture"}
                for i, truth in enumerate(truths)]
        calls = {method: 0 for method in benchmark.METHODS}

        def identify(image, *, method, **kwargs):
            status, ranked = predictions[method][calls[method]]
            calls[method] += 1
            return {"status": status, "candidates": [{"card_id": card} for card in ranked],
                    "mode": "cropped", "ocr": {}, "model_versions": {},
                    "before_ocr": [], "corners": None, "reason_codes": []}

        monkeypatch.setattr(benchmark, "identify", identify)
        output = tmp_path / "run"
        summary = benchmark.evaluate(rows, output)
        assert json.loads((output / "summary.json").read_text()) == summary
        return summary, output

    return run


def test_confusion_counts_are_per_method_and_include_unknown_and_reject(benchmark_run):
    predictions = {method: [("needs_confirmation", ["a"]),
                            ("needs_confirmation", ["a", "b"]),
                            ("unknown", ["a"]),
                            ("needs_confirmation", ["b"]),
                            ("unknown", [])]
                   for method in benchmark.METHODS}
    predictions["phash"][0] = ("needs_confirmation", ["b", "a"])
    summary, output = benchmark_run(["a", "b", "a", None, None], predictions)
    for method in benchmark.METHODS:
        metrics = summary["methods"][method]
        matrix = metrics["confusion_matrix"]
        assert matrix["ground_truth_labels"] == ["a", "b", {"kind": "unknown_or_reject"}]
        assert matrix["predicted_labels"] == ["a", "b", {"kind": "unknown_or_reject"}]
        assert matrix["counts"] == ([[0, 1, 1], [1, 0, 0], [0, 1, 1]] if method == "phash"
                                    else [[1, 0, 1], [1, 0, 0], [0, 1, 1]])
        assert sum(map(sum, matrix["counts"])) == 5
        assert metrics["top1"] == (1 / 3 if method == "phash" else 2 / 3)
        assert metrics["unknown_false_accept_rate"] == 0.5
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(results) == 20
    assert all(result["ranked_ids"] == ["a"] for result in results if result["sample_id"] == "2")
    assert (output / "results.csv").read_text().splitlines()[0] == "sample_id,ground_truth,method,ranked_ids,status,latency_ms"


@pytest.mark.parametrize("truth", ["a", None])
def test_confusion_handles_single_class_and_no_candidates(benchmark_run, truth):
    summary, _ = benchmark_run([truth], {method: [("needs_confirmation", [])]
                                        for method in benchmark.METHODS})
    for metrics in summary["methods"].values():
        assert metrics["confusion_matrix"] == {
            "ground_truth_labels": ["a", {"kind": "unknown_or_reject"}] if truth is not None else [{"kind": "unknown_or_reject"}],
            "predicted_labels": [{"kind": "unknown_or_reject"}],
            "counts": [[1], [0]] if truth is not None else [[1]],
        }
        assert metrics["top1"] == (0 if truth is not None else None)
        assert metrics["mrr_at_3"] == (0 if truth is not None else None)
        assert metrics["top3"] == (0 if truth is not None else None)
