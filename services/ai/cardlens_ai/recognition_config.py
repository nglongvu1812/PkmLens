"""Conservative live recognition thresholds, adjustable through environment variables."""
from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class RecognitionPolicy:
    min_similarity: float = 0.68
    open_set_min_similarity: float = 0.55
    min_margin: float = 0.035
    consensus_frames: int = 3
    min_sharpness: float = 12.0
    min_brightness: float = 22.0
    max_glare: float = 0.55
    card_change_ratio: float = 0.42
    signature_timeout_seconds: float = 2.0
    recognition_interval_seconds: float = 0.25
    best_frame_improvement_ratio: float = 1.4
    unconfirmed_refresh_seconds: float = 12.0
    best_frame_max_age_seconds: float = 2.0
    pose_corner_smoothing_alpha: float = 0.5
    pose_max_corner_jump_ratio: float = 0.2
    pose_min_keypoint_confidence: float = 0.5
    pose_min_area_ratio: float = 0.002
    pose_min_edge_pixels: float = 8.0

    def __post_init__(self):
        self.validate()

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        values = {
            "min_similarity": float(env.get("PKMLENS_RECOGNITION_MIN_SIMILARITY", "0.68")),
            "open_set_min_similarity": float(env.get("PKMLENS_OPEN_SET_MIN_SIMILARITY", "0.55")),
            "min_margin": float(env.get("PKMLENS_RECOGNITION_MIN_MARGIN", "0.035")),
            "consensus_frames": int(env.get("PKMLENS_RECOGNITION_CONSENSUS_FRAMES", "3")),
            "min_sharpness": float(env.get("PKMLENS_IMAGE_MIN_SHARPNESS", "12")),
            "min_brightness": float(env.get("PKMLENS_IMAGE_MIN_BRIGHTNESS", "22")),
            "max_glare": float(env.get("PKMLENS_IMAGE_MAX_GLARE", "0.55")),
            "card_change_ratio": float(env.get("PKMLENS_CARD_CHANGE_RATIO", "0.42")),
            "signature_timeout_seconds": float(env.get("PKMLENS_TRACK_SIGNATURE_TIMEOUT_SECONDS", "2")),
            "recognition_interval_seconds": float(env.get("PKMLENS_RECOGNITION_INTERVAL_SECONDS", "0.25")),
            "best_frame_improvement_ratio": float(env.get("PKMLENS_BEST_FRAME_IMPROVEMENT_RATIO", "1.4")),
            "unconfirmed_refresh_seconds": float(env.get("PKMLENS_UNCONFIRMED_REFRESH_SECONDS", "12")),
            "best_frame_max_age_seconds": float(env.get("PKMLENS_BEST_FRAME_MAX_AGE_SECONDS", "2")),
            "pose_corner_smoothing_alpha": float(env.get("PKMLENS_POSE_CORNER_SMOOTHING_ALPHA", "0.5")),
            "pose_max_corner_jump_ratio": float(env.get("PKMLENS_POSE_MAX_CORNER_JUMP_RATIO", "0.2")),
            "pose_min_keypoint_confidence": float(env.get("PKMLENS_POSE_MIN_KEYPOINT_CONFIDENCE", "0.5")),
            "pose_min_area_ratio": float(env.get("PKMLENS_POSE_MIN_AREA_RATIO", "0.002")),
            "pose_min_edge_pixels": float(env.get("PKMLENS_POSE_MIN_EDGE_PIXELS", "8")),
        }
        return cls(**values)

    def validate(self):
        policy = self
        numeric = (policy.min_similarity, policy.open_set_min_similarity, policy.min_margin,
                   policy.min_sharpness, policy.min_brightness, policy.max_glare,
                   policy.card_change_ratio, policy.signature_timeout_seconds,
                   policy.recognition_interval_seconds, policy.best_frame_improvement_ratio,
                   policy.unconfirmed_refresh_seconds, policy.consensus_frames,
                   policy.best_frame_max_age_seconds, policy.pose_corner_smoothing_alpha,
                   policy.pose_max_corner_jump_ratio, policy.pose_min_keypoint_confidence,
                   policy.pose_min_area_ratio, policy.pose_min_edge_pixels)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("recognition_threshold_not_finite")
        if not 0 <= policy.min_similarity <= 1:
            raise ValueError("min_similarity_out_of_range")
        if not 0 <= policy.open_set_min_similarity <= 1:
            raise ValueError("open_set_min_similarity_out_of_range")
        if not 0 <= policy.card_change_ratio <= 1:
            raise ValueError("card_change_ratio_out_of_range")
        if policy.signature_timeout_seconds <= 0:
            raise ValueError("signature_timeout_seconds_out_of_range")
        if policy.recognition_interval_seconds <= 0 or policy.unconfirmed_refresh_seconds <= 0:
            raise ValueError("recognition_interval_out_of_range")
        if policy.best_frame_improvement_ratio < 1:
            raise ValueError("best_frame_improvement_ratio_out_of_range")
        if not 0 <= policy.min_margin <= 1:
            raise ValueError("min_margin_out_of_range")
        if not isinstance(policy.consensus_frames, int) or isinstance(policy.consensus_frames, bool) or not 1 <= policy.consensus_frames <= 10:
            raise ValueError("consensus_frames_out_of_range")
        if policy.min_sharpness < 0:
            raise ValueError("min_sharpness_out_of_range")
        if not 0 <= policy.min_brightness <= 255:
            raise ValueError("min_brightness_out_of_range")
        if not 0 <= policy.max_glare <= 1:
            raise ValueError("max_glare_out_of_range")
        if policy.best_frame_max_age_seconds <= 0:
            raise ValueError("best_frame_max_age_seconds_out_of_range")
        if not 0 < policy.pose_corner_smoothing_alpha <= 1:
            raise ValueError("pose_corner_smoothing_alpha_out_of_range")
        if not 0 < policy.pose_max_corner_jump_ratio <= 1:
            raise ValueError("pose_max_corner_jump_ratio_out_of_range")
        if not 0 <= policy.pose_min_keypoint_confidence <= 1:
            raise ValueError("pose_min_keypoint_confidence_out_of_range")
        if not 0 < policy.pose_min_area_ratio < 1:
            raise ValueError("pose_min_area_ratio_out_of_range")
        if policy.pose_min_edge_pixels <= 0:
            raise ValueError("pose_min_edge_pixels_out_of_range")


DEFAULT_POLICY = RecognitionPolicy.from_env()
