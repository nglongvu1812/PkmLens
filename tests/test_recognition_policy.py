import pytest

from cardlens_ai.recognition_config import RecognitionPolicy
from cardlens_ai.live import TrackState


def test_policy_reads_adjustable_thresholds_from_environment():
    policy = RecognitionPolicy.from_env({
        "PKMLENS_RECOGNITION_MIN_SIMILARITY": "0.91",
        "PKMLENS_OPEN_SET_MIN_SIMILARITY": "0.62",
        "PKMLENS_RECOGNITION_MIN_MARGIN": "0.08",
        "PKMLENS_RECOGNITION_CONSENSUS_FRAMES": "4",
        "PKMLENS_IMAGE_MIN_SHARPNESS": "42",
        "PKMLENS_IMAGE_MIN_BRIGHTNESS": "40",
        "PKMLENS_IMAGE_MAX_GLARE": "0.3",
        "PKMLENS_CARD_CHANGE_RATIO": "0.4",
        "PKMLENS_TRACK_SIGNATURE_TIMEOUT_SECONDS": "3.5",
        "PKMLENS_RECOGNITION_INTERVAL_SECONDS": "2.5",
        "PKMLENS_BEST_FRAME_IMPROVEMENT_RATIO": "1.8",
        "PKMLENS_UNCONFIRMED_REFRESH_SECONDS": "15",
    })

    assert policy.min_similarity == 0.91
    assert policy.open_set_min_similarity == 0.62
    assert policy.min_margin == 0.08
    assert policy.consensus_frames == 4
    assert policy.min_sharpness == 42
    assert policy.min_brightness == 40
    assert policy.max_glare == 0.3
    assert policy.card_change_ratio == 0.4
    assert policy.signature_timeout_seconds == 3.5
    assert policy.recognition_interval_seconds == 2.5
    assert policy.best_frame_improvement_ratio == 1.8
    assert policy.unconfirmed_refresh_seconds == 15


def test_policy_rejects_out_of_range_thresholds():
    with pytest.raises(ValueError, match="min_similarity"):
        RecognitionPolicy.from_env({"PKMLENS_RECOGNITION_MIN_SIMILARITY": "1.2"})
    with pytest.raises(ValueError, match="open_set_min_similarity"):
        RecognitionPolicy.from_env({"PKMLENS_OPEN_SET_MIN_SIMILARITY": "-0.1"})
    with pytest.raises(ValueError, match="not_finite"):
        RecognitionPolicy.from_env({"PKMLENS_RECOGNITION_MIN_MARGIN": "nan"})


def test_track_requires_calibrated_policy_thresholds_and_consensus():
    policy = RecognitionPolicy(min_similarity=0.9, min_margin=0.08, consensus_frames=2)
    state = TrackState()
    state.observe(None, 1)
    result = {"candidates": [
        {"card_id": "set:7:en", "similarity": 0.92, "evidence": {"number_match": True, "total_match": True}},
        {"card_id": "other:7:en", "similarity": 0.80, "evidence": {}},
    ], "ocr": {"number": "7"}}

    state.accept(result, policy)
    assert state.status == "stabilizing" and state.votes == 1
    state.accept(result, policy)
    assert state.status == "needs_confirmation" and state.votes == 2

    result["candidates"][0]["similarity"] = 0.89
    state.accept(result, policy)
    assert state.status == "unknown" and state.votes == 0


def test_image_quality_rejects_blurry_and_glared_frames_using_policy():
    import numpy as np
    from PIL import Image
    from cardlens_ai.detection import quality

    policy = RecognitionPolicy(min_sharpness=1, min_brightness=0, max_glare=0.5)
    blurry = quality(Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)), policy)
    glared = quality(Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8)), policy)

    assert blurry["usable"] is False and "blurred" in blurry["reasons"]
    assert glared["usable"] is False and "glare" in glared["reasons"]


def test_confirmed_track_reuses_identity_until_card_signature_changes():
    import numpy as np
    policy = RecognitionPolicy(consensus_frames=2)
    state = TrackState(signature=np.ones(63, dtype=np.uint8), votes=2, last_recognized=10, best_quality=50)

    assert state.should_recognize(now=11, sharpness=100, policy=policy) is False
    state.observe(np.zeros(63, dtype=np.uint8), 12, policy)
    assert state.should_recognize(now=13, sharpness=50, policy=policy) is True
