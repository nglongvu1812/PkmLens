"""Runtime contracts only: stub pose predictions are not model-accuracy evidence."""
import asyncio
import hashlib
import importlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from ultralytics.engine.results import Boxes

from cardlens_ai import detection
from cardlens_ai.recognition_config import RecognitionPolicy

live_module = importlib.import_module("cardlens_ai.live")


def identity(card="set:7:en"):
    return {"candidates": [{"card_id": card, "similarity": .95,
                            "evidence": {"number_match": True, "total_match": True}}],
            "ocr": {"number": "7"}}


@pytest.mark.parametrize("wrapped", [False, True])
def test_real_bytetrack_accepts_numpy_boxes_and_preserves_detection_indices(wrapped):
    rows = np.array([[1, 1, 30, 50, .2, 0], [60, 10, 110, 90, .95, 0]], dtype=np.float32)
    source = Boxes(rows, (100, 120)) if wrapped else rows
    adapted = detection.tracker_input(source, (100, 120))
    assert isinstance(adapted.data, np.ndarray)
    if wrapped:
        assert adapted is source
    tracker = detection.new_tracker()
    tracked = tracker.update(adapted)
    assert tracked.shape == (1, 8)
    assert tracked[0, -1] == 1
    moved = rows.copy()
    moved[1, :4] += [8, 0, 8, 0]
    tracked = tracker.update(detection.tracker_input(moved, (100, 120)))
    assert tracked[0, -1] == 1
    assert tracked[0, 4] == 1
    assert tracker.update(detection.tracker_input(np.empty((0, 6)), (100, 120))).size == 0


def test_torch_backed_boxes_are_converted_before_real_tracker():
    import torch
    source = Boxes(torch.tensor([[10, 10, 60, 90, .9, 0]]), (100, 100))
    adapted = detection.tracker_input(source, (100, 100))
    assert isinstance(adapted.data, np.ndarray)
    assert detection.new_tracker().update(adapted).shape == (1, 8)


def test_confirmed_signature_anchor_catches_gradual_replacement_and_clears_best_frame():
    policy = RecognitionPolicy(consensus_frames=2)
    state = live_module.TrackState()
    signature = np.zeros(64, dtype=np.uint8)
    state.observe(signature, 1, policy)
    state.accept(identity(), policy)
    state.accept(identity(), policy)
    generation = state.generation
    for changed_bits in (8, 16, 24):
        signature[:changed_bits] = 1
        assert not state.observe(signature, 1 + changed_bits / 100, policy)
        assert not state.should_recognize(100, 10000, policy)
    signature[:32] = 1
    assert state.observe(signature, 1.4, policy)
    assert state.generation != generation
    assert state.votes == 0 and state.candidates == [] and state.ocr == {}
    assert state.best_frame is None and state.identity_signature is None
    assert state.best_quality == 0


def test_best_frame_slot_is_bounded_fresh_and_cleared_at_confirmation():
    policy = RecognitionPolicy(consensus_frames=1, best_frame_max_age_seconds=1)
    state = live_module.TrackState()
    signature = np.zeros(63, dtype=np.uint8)
    state.observe(signature, 1, policy)
    for score in range(1, 30):
        state.offer_frame(Image.new("RGB", (8, 8), (score, 0, 0)), signature, score, 1 + score / 100, policy)
    chosen = state.take_best_frame(1.4, policy)
    assert chosen.sharpness == 29 and chosen.image.getpixel((0, 0))[0] == 29
    assert state.best_frame is None
    state.offer_frame(Image.new("RGB", (8, 8)), signature, 50, 2, policy)
    assert state.take_best_frame(3.1, policy) is None
    state.offer_frame(Image.new("RGB", (8, 8)), signature, 1, 4, policy)
    state.accept(identity(), policy, signature=signature)
    assert state.best_frame is None
    state.offer_frame(Image.new("RGB", (8, 8)), signature, 100, 5, policy)
    assert state.best_frame is None


def test_per_track_pose_smoothing_rejects_jumps_and_invalid_geometry():
    policy = RecognitionPolicy(pose_corner_smoothing_alpha=.5, pose_max_corner_jump_ratio=.2)
    state = live_module.TrackState()
    corners = np.float32([[10, 10], [90, 10], [90, 150], [10, 150]])
    assert np.allclose(state.smooth_corners(corners, 300, 300, 1, policy), corners)
    assert np.allclose(state.smooth_corners(corners + 4, 300, 300, 1.1, policy), corners + 2)
    with pytest.raises(ValueError, match="pose_corner_jump"):
        state.smooth_corners(corners + 100, 300, 300, 1.2, policy)
    with pytest.raises(ValueError):
        state.smooth_corners([[10, 10], [90, 10], [20, 20], [10, 150]], 300, 300, 1.3, policy)
    with pytest.raises(ValueError):
        state.smooth_corners(np.full((4, 2), np.nan), 300, 300, 1.4, policy)
    assert live_module.TrackState().smoothed_corners is None
    assert np.allclose(state.smooth_corners(corners + 100, 300, 300, 4, policy), corners + 100)


@pytest.mark.parametrize("name,value", [
    ("PKMLENS_POSE_CORNER_SMOOTHING_ALPHA", "0"),
    ("PKMLENS_POSE_MAX_CORNER_JUMP_RATIO", "nan"),
    ("PKMLENS_POSE_MIN_KEYPOINT_CONFIDENCE", "1.1"),
    ("PKMLENS_POSE_MIN_AREA_RATIO", "0"),
    ("PKMLENS_POSE_MIN_EDGE_PIXELS", "-1"),
    ("PKMLENS_BEST_FRAME_MAX_AGE_SECONDS", "0"),
])
def test_temporal_policy_rejects_invalid_values(name, value):
    with pytest.raises(ValueError):
        RecognitionPolicy.from_env({name: value})


def pose_registry(tmp_path, monkeypatch, **extra):
    checkpoint = tmp_path / "pose.pt"
    checkpoint.write_bytes(b"stub weights, not a trained checkpoint")
    info = {"id": "pose", "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "validation_passed": True, "task": "pose", "classes": ["pokemon_card"],
            "keypoints": ["TL", "TR", "BR", "BL"], **extra}
    monkeypatch.setattr(detection, "DATA", tmp_path)
    (tmp_path / "models").mkdir(exist_ok=True)
    registry_path = tmp_path / "models/pose-detectors.json"
    registry_path.write_text(json.dumps({"active": "pose", "models": [info]}))
    return checkpoint, info, registry_path


def test_pose_metadata_is_verified_allowlisted_and_agrees_with_health(tmp_path, monkeypatch):
    from cardlens_ai import main
    checkpoint, info, _ = pose_registry(tmp_path, monkeypatch,
                                        notes="D:/private/path", training={"path": "D:/private/path"})
    monkeypatch.setenv("AI_INTERNAL_KEY", "test")
    monkeypatch.setattr(detection, "registry", lambda: {"active": None, "models": []})
    result = main.detectors("test")
    assert result["pose_active"] == "pose" and main.health()["pose_ready"]
    assert "private" not in json.dumps(result) and "checkpoint" not in json.dumps(result)
    assert "cuda" in result and "device" in result
    checkpoint.write_bytes(b"tampered")
    result = main.detectors("test")
    assert result["pose_active"] is None and result["pose_models"] == []
    assert not main.health()["pose_ready"]


@pytest.mark.parametrize("bad_score", [.1, 1.2, float("nan")])
def test_real_boxes_pose_stub_gates_confidence_per_card_and_reloads_path(tmp_path, monkeypatch, bad_score):
    import torch
    import ultralytics
    from types import SimpleNamespace
    checkpoint, info, registry_path = pose_registry(tmp_path, monkeypatch)
    real_read_bytes = Path.read_bytes
    checkpoint_reads = []
    def counted_read(path):
        if path.suffix == ".pt":checkpoint_reads.append(path)
        return real_read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", counted_read)
    rows = np.float32([[10, 10, 90, 150, .9, 0], [110, 10, 190, 150, .9, 0]])
    points = torch.tensor([[[10, 10, .9], [90, 10, .9], [90, 150, .9], [10, 150, .9]],
                           [[110, 10, .9], [190, 10, bad_score], [190, 150, .9], [110, 150, .9]]])
    from ultralytics.engine.results import Keypoints
    result = SimpleNamespace(boxes=Boxes(torch.from_numpy(rows), (200, 200)),
                             keypoints=Keypoints(points, (200, 200)))
    loaded = []
    class Model:
        task = "pose"
        names = {0: "pokemon_card"}
        model = SimpleNamespace(kpt_shape=(4, 3), args={"task": "pose"})
        def predict(self, *args, **kwargs): return [result]
    monkeypatch.setattr(ultralytics, "YOLO", lambda path, **kwargs: loaded.append(path) or Model())
    monkeypatch.setattr(detection, "_pose_model", None)
    monkeypatch.setattr(detection, "_pose_loaded", None)
    monkeypatch.setenv("DETECTOR_DEVICE", "cpu")
    boxes, corners, _ = detection.detect_pose(Image.new("RGB", (200, 200)))
    assert boxes.shape == (2, 6) and np.isfinite(corners[0]).all()
    assert not np.isfinite(corners[1]).all()
    detection.detect_pose(Image.new("RGB", (200, 200)))
    assert len(loaded) == 1 and checkpoint_reads == [checkpoint]
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(checkpoint.read_bytes())
    info["checkpoint"] = str(replacement)
    registry_path.write_text(json.dumps({"active": "pose", "models": [info]}))
    detection.detect_pose(Image.new("RGB", (200, 200)))
    assert loaded == [str(checkpoint), str(replacement)]
    replacement.write_bytes(b"tampered after loading")
    with pytest.raises(ValueError, match="checksum"):
        detection.detect_pose(Image.new("RGB", (200, 200)))


class Socket:
    headers = {"x-cardlens-key": "test"}
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.sent = []
    async def accept(self): pass
    async def close(self, **kwargs): pass
    async def receive(self): return await self.incoming.get()
    async def send_json(self, value):
        self.sent.append(value)
        self.outgoing.put_nowait(value)
    async def message(self, kind):
        while True:
            value = await asyncio.wait_for(self.outgoing.get(), 5)
            if value["type"] == "error":
                raise AssertionError(value)
            if value["type"] == kind:
                return value
    def frame(self, value): self.incoming.put_nowait({"type": "websocket.receive", "bytes": bytes([value])})
    def configure(self): self.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({"type": "configure"})})
    def disconnect(self): self.incoming.put_nowait({"type": "websocket.disconnect"})


@pytest.fixture
def live_runtime(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_KEY", "test")
    monkeypatch.setattr(live_module, "active_sessions", 0)
    monkeypatch.setattr(live_module, "recognition_gate", asyncio.Semaphore(1))
    monkeypatch.setattr(live_module, "registry", lambda: {"active": "bbox", "models": []})
    monkeypatch.setattr(live_module, "verified_pose_metadata", lambda: {"pose_ready": False, "pose_active": None, "pose_models": []})
    monkeypatch.setattr(live_module, "unload", lambda: None)
    monkeypatch.setattr(live_module, "decode", lambda raw: Image.new("RGB", (200, 200), (raw[0], 0, 0)))
    rows = np.float32([[10, 10, 90, 150, .9, 0]])
    monkeypatch.setattr(live_module, "detect", lambda *args: (Boxes(rows.copy(), (200, 200)), {"id": "bbox"}))
    def crop(image, box, corners, policy):
        value = image.getpixel((0, 0))[0]
        return image, None, {"usable": True, "sharpness": value, "reasons": []}, np.zeros(63, dtype=np.uint8), "bbox_fallback"
    monkeypatch.setattr(live_module, "prepare_crop", crop)
    monkeypatch.setattr(live_module, "DEFAULT_POLICY", RecognitionPolicy(consensus_frames=1))
    monkeypatch.setattr(live_module, "recognize", lambda image: identity())
    return Socket()


@pytest.mark.asyncio
async def test_configure_discards_inflight_detect_and_queued_frames(live_runtime, monkeypatch):
    ws = live_runtime
    entered, release = threading.Event(), threading.Event()
    original = live_module.detect
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(live_module, "detect", blocked)
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.frame(1)
        assert await asyncio.to_thread(entered.wait, 5)
        ws.frame(2)
        ws.configure()
        configured = await ws.message("configured")
        assert configured["configuration_revision"] == 1 and configured["tracks"] == []
        release.set()
        ws.frame(3)
        frame = await ws.message("frame")
        assert frame["sequence"] == 3 and frame["configuration_revision"] == 1
        assert not [m for m in ws.sent if m["type"] == "frame" and m["sequence"] < 3]
    finally:
        release.set()
        ws.disconnect()
        await asyncio.wait_for(task, 5)
    assert live_module.active_sessions == 0


@pytest.mark.asyncio
async def test_waiting_recognition_uses_best_recent_crop_and_confirmed_cache(live_runtime, monkeypatch):
    ws = live_runtime
    gate = asyncio.Semaphore(0)
    monkeypatch.setattr(live_module, "recognition_gate", gate)
    recognized = []
    monkeypatch.setattr(live_module, "recognize", lambda image: recognized.append(image.getpixel((0, 0))[0]) or identity())
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.frame(1)
        await ws.message("frame")
        ws.frame(20)
        await ws.message("frame")
        gate.release()
        result = await ws.message("identity")
        assert result["configuration_revision"] == 0 and recognized == [20]
        ws.frame(30)
        await ws.message("frame")
        assert recognized == [20]
    finally:
        ws.disconnect()
        await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_configure_and_disconnect_do_not_leak_running_thread_result(live_runtime, monkeypatch):
    ws = live_runtime
    entered, release = threading.Event(), threading.Event()
    def recognize(image):
        entered.set()
        assert release.wait(5)
        return identity("old")
    monkeypatch.setattr(live_module, "recognize", recognize)
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.frame(1)
        await ws.message("frame")
        assert await asyncio.to_thread(entered.wait, 5)
        ws.configure()
        await ws.message("configured")
        release.set()
        ws.frame(2)
        await ws.message("frame")
        result = await ws.message("identity")
        assert result["configuration_revision"] == 1
        assert all(m["configuration_revision"] == 1 for m in ws.sent if m["type"] == "identity")
    finally:
        release.set()
        ws.disconnect()
        await asyncio.wait_for(task, 5)
    assert live_module.active_sessions == 0 and not live_module.recognition_gate.locked()


@pytest.mark.asyncio
async def test_tracker_initialization_failure_releases_session_counter(live_runtime, monkeypatch):
    def fail(*args): raise RuntimeError("initialization test")
    monkeypatch.setattr(live_module, "new_tracker", fail)
    await live_module.live(live_runtime)
    assert live_module.active_sessions == 0

@pytest.mark.parametrize("task,shape,names,allowed", [
    ("pose", (4, 3), {0: "pokemon_card"}, True),
    ("detect", (4, 3), {0: "pokemon_card"}, False),
    ("pose", (17, 3), {0: "pokemon_card"}, False),
    ("pose", (4, 3), {0: "person"}, False),
])
def test_activation_helper_checks_underlying_model_without_touching_inference_cache(tmp_path, monkeypatch, task, shape, names, allowed):
    import ultralytics
    from types import SimpleNamespace
    checkpoint = tmp_path / "local.pt"
    checkpoint.write_bytes(b"stub activation contract only")
    model = SimpleNamespace(task="pose", names=names,
                            model=SimpleNamespace(task=task, kpt_shape=shape))
    monkeypatch.setattr(ultralytics, "YOLO", lambda path, **kwargs: model)
    cached = object()
    monkeypatch.setattr(detection, "_pose_model", cached)
    monkeypatch.setattr(detection, "_pose_loaded", ("unchanged",))
    if allowed:
        assert detection.verify_pose_checkpoint(checkpoint) == {
            "task": "pose", "classes": ["pokemon_card"], "keypoints": ["TL", "TR", "BR", "BL"]}
    else:
        with pytest.raises(ValueError):
            detection.verify_pose_checkpoint(checkpoint)
    assert detection._pose_model is cached and detection._pose_loaded == ("unchanged",)


def test_bbox_crop_preserves_explicit_fallback_label():
    image = Image.new("RGB", (200, 200), (100, 100, 100))
    _, ordered, _, _, correction = live_module.prepare_crop(image, [20, 20, 180, 180], None)
    assert ordered is None and correction == "bbox_fallback"


@pytest.mark.asyncio
async def test_disconnect_keeps_recognition_slot_until_actual_thread_finishes(live_runtime, monkeypatch):
    ws = live_runtime
    entered, release = threading.Event(), threading.Event()
    def recognize(image):
        entered.set()
        assert release.wait(5)
        return identity()
    monkeypatch.setattr(live_module, "recognize", recognize)
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.frame(1)
        await ws.message("frame")
        assert await asyncio.to_thread(entered.wait, 5)
        ws.disconnect()
        # Yield until receive/process cancellation is reached, without a timing sleep.
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done() and live_module.recognition_gate.locked()
        assert live_module.active_sessions == 1
        release.set()
        await asyncio.wait_for(task, 5)
        assert live_module.active_sessions == 0 and not live_module.recognition_gate.locked()
        assert not [m for m in ws.sent if m["type"] == "identity"]
    finally:
        release.set()
        if not task.done():
            ws.disconnect()
            await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_pose_row_index_not_nearest_box_and_bad_card_is_rejected_individually(live_runtime, monkeypatch):
    ws = live_runtime
    monkeypatch.setattr(live_module, "verified_pose_metadata", lambda: {
        "pose_ready": True, "pose_active": "pose", "pose_models": [{"id": "pose"}]})
    rows = np.float32([[1, 1, 30, 50, .2, 0], [60, 10, 110, 150, .95, 0], [120, 10, 190, 150, .95, 0]])
    corners = np.float32([[[1, 1], [30, 1], [30, 50], [1, 50]],
                           [[60, 10], [110, 10], [110, 150], [60, 150]],
                           [[120, 10], [190, 10], [125, 20], [120, 150]]])
    monkeypatch.setattr(live_module, "detect_pose", lambda *args: (rows, corners, {"id": "pose"}))
    monkeypatch.setattr(live_module, "prepare_crop", original_prepare_crop)
    # Use real ByteTrack filtering/indexing, then emulate a filtered Kalman box nearer row 0.
    make_tracker = detection.new_tracker
    class DriftingTracker:
        def __init__(self, confidence=.45): self.tracker = make_tracker(confidence)
        def update(self, boxes, image):
            tracked = self.tracker.update(boxes, image)
            if len(tracked): tracked[0, :4] = rows[0, :4]
            return tracked
    monkeypatch.setattr(live_module, "new_tracker", DriftingTracker)
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({
            "type": "configure", "use_pose": True, "pose_model": "pose"})})
        await ws.message("configured")
        ws.frame(1)
        frame = await ws.message("frame")
        assert len(frame["tracks"]) == 2
        valid, invalid = frame["tracks"]
        assert np.allclose(valid["quadrilateral"], corners[1] / 200)
        assert valid["correction"] == "yolo26_pose"
        assert invalid["status"] == "reject" and invalid["correction"] == "pose_rejected"
        assert invalid["quadrilateral"] is None and invalid["candidates"] == []
    finally:
        ws.disconnect()
        await asyncio.wait_for(task, 5)


original_prepare_crop = live_module.prepare_crop

@pytest.mark.asyncio
async def test_configure_invalidates_before_checkpoint_boundary_io(live_runtime, monkeypatch):
    ws = live_runtime
    entered, release = threading.Event(), threading.Event()
    checking, checked = threading.Event(), threading.Event()
    def recognize(image):
        entered.set()
        assert release.wait(5)
        return identity()
    checks = []
    def metadata():
        checks.append(True)
        if len(checks) > 1:
            checking.set()
            assert checked.wait(5)
        return {"pose_ready": False, "pose_active": None, "pose_models": []}
    monkeypatch.setattr(live_module, "recognize", recognize)
    monkeypatch.setattr(live_module, "verified_pose_metadata", metadata)
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.frame(1)
        await ws.message("frame")
        assert await asyncio.to_thread(entered.wait, 5)
        ws.configure()
        assert await asyncio.to_thread(checking.wait, 5)
        release.set()
        # Complete a following worker call to let the recognition coroutine run.
        await live_module.run_thread(lambda: None)
        for _ in range(20): await asyncio.sleep(0)
        assert not [m for m in ws.sent if m["type"] == "identity"]
        checked.set()
        await ws.message("configured")
    finally:
        release.set();checked.set()
        ws.disconnect()
        await asyncio.wait_for(task, 5)


def test_checkpoint_verifier_loads_actual_untrained_local_pose_architecture(tmp_path):
    """Random weights prove loader/schema checks, never pose accuracy or training readiness."""
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import YAML
    config = YAML.load(str(Path(ultralytics.__file__).parent / "cfg/models/26/yolo26-pose.yaml"))
    config.update(nc=1, kpt_shape=[4, 3], scale="n")
    config_path = tmp_path / "card-pose.yaml"
    YAML.save(str(config_path), config)
    model = YOLO(str(config_path), task="pose")
    model.model.names = {0: "pokemon_card"}
    checkpoint = tmp_path / "untrained-card-pose.pt"
    model.save(str(checkpoint))
    assert detection.verify_pose_checkpoint(checkpoint)["keypoints"] == ["TL", "TR", "BR", "BL"]

@pytest.mark.asyncio
async def test_configuration_id_echo_and_invalid_pose_keeps_valid_session(live_runtime):
    ws = live_runtime
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({
            "type": "configure", "configuration_id": 3})})
        configured = await ws.message("configured")
        assert configured["configuration_id"] == 3 and configured["configuration_revision"] == 1
        ws.frame(1)
        frame = await ws.message("frame")
        recognized = await ws.message("identity")
        assert frame["configuration_id"] == recognized["configuration_id"] == 3
        ws.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({
            "type": "configure", "configuration_id": 4, "use_pose": True, "pose_model": "inactive"})})
        error = await asyncio.wait_for(ws.outgoing.get(), 5)
        assert error["type"] == "error" and error["configuration_id"] == 4
        assert error["code"] == "pose_detector_not_active"
        ws.frame(2)
        frame = await ws.message("frame")
        assert frame["configuration_id"] == 3 and frame["configuration_revision"] == 1
        assert frame["tracks"][0]["votes"] == 1
        ws.configure()
        configured = await ws.message("configured")
        assert configured["configuration_id"] is None and configured["configuration_revision"] == 2
    finally:
        ws.disconnect()
        await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [
    {"confidence": "nan"}, {"confidence": None}, {"size": 320.9},
    {"use_pose": "false"}, {"configuration_id": True}, {"configuration_id": -1},
])
async def test_invalid_config_reports_error_without_closing_socket(live_runtime, config):
    ws = live_runtime
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({"type": "configure", **config})})
        error = await asyncio.wait_for(ws.outgoing.get(), 5)
        assert error["type"] == "error" and error["code"] == "invalid_config"
        ws.configure()
        ack = await ws.message("configured")
        assert ack["configuration_revision"] == 1 and ack["configuration_id"] is None
        ws.frame(1)
        assert (await ws.message("frame"))["configuration_revision"] == 1
    finally:
        ws.disconnect()
        await asyncio.wait_for(task, 5)

@pytest.mark.asyncio
async def test_card_change_during_recognition_discards_old_identity(live_runtime, monkeypatch):
    ws = live_runtime
    entered, release = threading.Event(), threading.Event()
    recognized = []
    def recognize(image):
        value = image.getpixel((0, 0))[0]
        recognized.append(value)
        if value == 1:
            entered.set()
            assert release.wait(5)
        return identity(str(value))
    def crop(image, box, corners, policy):
        value = image.getpixel((0, 0))[0]
        return image, None, {"usable": True, "sharpness": value, "reasons": []}, np.full(63, value > 1, dtype=np.uint8), "bbox_fallback"
    monkeypatch.setattr(live_module, "recognize", recognize)
    monkeypatch.setattr(live_module, "prepare_crop", crop)
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.frame(1)
        first = await ws.message("frame")
        assert await asyncio.to_thread(entered.wait, 5)
        ws.frame(2)
        second = await ws.message("frame")
        assert first["tracks"][0]["generation"] != second["tracks"][0]["generation"]
        release.set()
        result = await ws.message("identity")
        assert result["candidates"][0]["card_id"] == "2"
        assert recognized == [1, 2]
        assert len([m for m in ws.sent if m["type"] == "identity"]) == 1
    finally:
        release.set()
        ws.disconnect()
        await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_expired_waiting_crop_is_not_recognized(live_runtime, monkeypatch):
    ws = live_runtime
    gate = asyncio.Semaphore(0)
    monkeypatch.setattr(live_module, "recognition_gate", gate)
    from types import SimpleNamespace
    import time
    clock = [100.0]
    monkeypatch.setattr(live_module, "time", SimpleNamespace(monotonic=lambda: clock[0], perf_counter=time.perf_counter))
    monkeypatch.setattr(live_module, "DEFAULT_POLICY", RecognitionPolicy(best_frame_max_age_seconds=1))
    def crop(image, box, corners, policy):
        # Advance only the application clock; Windows monotonic resolution makes sub-ms sleeps flaky.
        clock[0] += 2
        return image, None, {"usable": True, "sharpness": 100, "reasons": []}, np.zeros(63, dtype=np.uint8), "bbox_fallback"
    monkeypatch.setattr(live_module, "prepare_crop", crop)
    recognized = []
    monkeypatch.setattr(live_module, "recognize", lambda image: recognized.append(True) or identity())
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.frame(1)
        await ws.message("frame")
        gate.release()
        await live_module.run_thread(lambda: None)
        for _ in range(20): await asyncio.sleep(0)
        assert recognized == []
    finally:
        ws.disconnect()
        await asyncio.wait_for(task, 5)

@pytest.mark.asyncio
async def test_identity_is_not_emitted_before_first_multicard_frame(live_runtime, monkeypatch):
    ws = live_runtime
    rows = np.float32([[10, 10, 90, 150, .9, 0], [110, 10, 190, 150, .9, 0]])
    monkeypatch.setattr(live_module, "detect", lambda *args: (Boxes(rows.copy(), (200, 200)), {"id": "bbox"}))
    cropped_second = threading.Event()
    def crop(image, box, corners, policy):
        if box[0] > 100:
            import time
            time.sleep(.05)
            cropped_second.set()
        return image, None, {"usable": True, "sharpness": 100, "reasons": []}, np.zeros(63, dtype=np.uint8), "bbox_fallback"
    monkeypatch.setattr(live_module, "prepare_crop", crop)
    task = asyncio.create_task(live_module.live(ws))
    try:
        await ws.message("ready")
        ws.frame(1)
        first = await asyncio.wait_for(ws.outgoing.get(), 5)
        assert first["type"] == "frame" and cropped_second.is_set()
        await ws.message("identity")
    finally:
        ws.disconnect()
        await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_websocket_accept_failure_does_not_leak_session_counter(live_runtime, monkeypatch):
    async def fail(): raise RuntimeError("accept failed")
    live_runtime.accept = fail
    with pytest.raises(RuntimeError, match="accept failed"):
        await live_module.live(live_runtime)
    assert live_module.active_sessions == 0

@pytest.mark.asyncio
async def test_session_limit_is_reserved_before_async_accept(live_runtime, monkeypatch):
    accepted = asyncio.Event()
    sockets = [Socket(), Socket(), Socket()]
    closed = []
    async def accept(): await accepted.wait()
    async def close(**kwargs): closed.append(kwargs["code"])
    for ws in sockets:
        ws.accept = accept
        ws.close = close
    tasks = [asyncio.create_task(live_module.live(ws)) for ws in sockets]
    try:
        for _ in range(20): await asyncio.sleep(0)
        assert live_module.active_sessions == 2 and closed == [1013]
    finally:
        accepted.set()
        for ws in sockets:ws.disconnect()
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
    assert live_module.active_sessions == 0

def test_bbox_failed_reload_keeps_last_verified_cache_usable(tmp_path, monkeypatch):
    import ultralytics
    from types import SimpleNamespace
    good_path, bad_path = tmp_path / "good.pt", tmp_path / "bad.pt"
    good_path.write_bytes(b"good local fixture")
    bad_path.write_bytes(b"invalid class fixture")
    def info(path, ident):
        return {"id": ident, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    records = {"good": (info(good_path, "good"), good_path), "bad": (info(bad_path, "bad"), bad_path)}
    monkeypatch.setattr(detection, "resolve_model", lambda ident: records[ident])
    class Model:
        names = {0: "pokemon_card"}
        def predict(self, *args, **kwargs):
            return [SimpleNamespace(boxes=Boxes(np.float32([[10, 10, 90, 150, .9, 0]]), (200, 200)))]
    monkeypatch.setattr(ultralytics, "YOLO", lambda path: Model() if path == str(good_path) else SimpleNamespace(names={0: "person"}))
    monkeypatch.setattr(detection, "_model", None)
    monkeypatch.setattr(detection, "_loaded", None)
    monkeypatch.setenv("DETECTOR_DEVICE", "cpu")
    detection.detect(Image.new("RGB", (200, 200)), "good")
    verified_model, signature = detection._model, detection._loaded
    with pytest.raises(ValueError, match="not_a_card"):
        detection.detect(Image.new("RGB", (200, 200)), "bad")
    assert detection._model is verified_model and detection._loaded == signature
    boxes, _ = detection.detect(Image.new("RGB", (200, 200)), "good")
    assert isinstance(boxes, Boxes)

@pytest.mark.parametrize("status", ["unknown", "reject", "rejected"])
def test_temporal_votes_never_override_pipeline_abstention(status):
    state = live_module.TrackState()
    result = identity()
    result.update(status=status, margin=.2, margin_score="fusion_score")
    for _ in range(4):state.accept(result)
    assert state.votes == 0 and state.status == "unknown"


@pytest.mark.parametrize("key", ["name_match", "set_code_match", "language_match"])
def test_temporal_votes_reject_all_explicit_ocr_conflicts(key):
    state = live_module.TrackState()
    result = identity()
    result["candidates"][0]["evidence"][key] = False
    for _ in range(4):state.accept(result)
    assert state.votes == 0 and state.status == "unknown"


def test_temporal_votes_use_measured_fused_margin_and_reject_missing_margin():
    state = live_module.TrackState()
    result = identity()
    result["candidates"].append({"card_id": "other", "similarity": .94, "evidence": {}})
    result.update(status="needs_confirmation", margin=.2, margin_score="fusion_score")
    for _ in range(3):state.accept(result)
    assert state.votes == 3 and state.status == "needs_confirmation"
    result["margin"] = None
    state.accept(result)
    assert state.votes == 0 and state.identity_signature is None


@pytest.mark.asyncio
async def test_queue_latency_uses_same_clock_as_frame_arrival(live_runtime, monkeypatch):
    from types import SimpleNamespace
    # Windows Python 3.12 uses different epochs/resolutions for these clocks.
    monkeypatch.setattr(live_module, "time", SimpleNamespace(monotonic=lambda: 100.0, perf_counter=lambda: 1000.0))
    task = asyncio.create_task(live_module.live(live_runtime))
    try:
        await live_runtime.message("ready")
        live_runtime.frame(1)
        result = await live_runtime.message("frame")
        assert result["timing_ms"]["queue"] == 0.0
        assert result["timing_ms"]["detect_track"] == 0.0
    finally:
        live_runtime.disconnect()
        await asyncio.wait_for(task, 5)
