import asyncio
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from cardlens_ai.live import TrackState, put_latest
from cardlens_core.main import app


def test_track_identity_requires_evidence_and_resets_on_card_change():
    state=TrackState()
    state.observe(np.zeros(63,dtype=np.uint8),1)
    result={'candidates':[{'card_id':'set:7:en','similarity':.9,'evidence':{'number_match':True,'total_match':True}},
                          {'card_id':'other:7:en','similarity':.7,'evidence':{}}],'ocr':{'number':'7'}}
    for _ in range(3):state.accept(result)
    assert state.status=='needs_confirmation' and state.votes==3
    generation=state.generation
    state.observe(np.ones(63,dtype=np.uint8),1.2)
    assert state.generation!=generation and not state.candidates and state.votes==0
    result['candidates'][0]['evidence']['number_match']=False
    for _ in range(4):state.accept(result)
    assert state.status=='unknown' and state.votes==0


def test_latest_queue_stays_bounded():
    queue=asyncio.Queue(maxsize=1)
    for i in range(100):put_latest(queue,i)
    assert queue.qsize()==1 and queue.get_nowait()==99


def test_replacing_a_card_discards_old_ocr_and_unknown_resets_identity_vote():
    state = TrackState()
    state.observe(np.zeros(63, dtype=np.uint8), 1)
    state.ocr = {'name': 'Previous card', 'number': '7'}
    state.observe(np.ones(63, dtype=np.uint8), 1.1)
    assert state.ocr == {}
    state.last_card, state.votes = 'old:7:en', 3
    state.accept({'candidates': [], 'ocr': {'name': 'Unknownmon'}})
    assert state.last_card is None and state.votes == 0 and state.status == 'unknown'


def test_live_ws_requires_session_and_origin():
    client=TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/api/v1/ws/scan',headers={'origin':'http://127.0.0.1:3000'}):pass
    client.post('/api/v1/session')
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/api/v1/ws/scan',headers={'origin':'https://outside.example'}):pass


@pytest.mark.asyncio
async def test_gateway_closes_cleanly_after_browser_has_disconnected(monkeypatch):
    from contextlib import asynccontextmanager
    from cardlens_core import live_scan
    import websockets.asyncio.client

    class Socket:
        headers = {'origin': 'http://127.0.0.1:5173'}
        async def accept(self): pass
        async def close(self, **kwargs): raise WebSocketDisconnect(code=1006)

    @asynccontextmanager
    async def disconnected(*args, **kwargs):
        raise WebSocketDisconnect(code=1006)
        yield

    monkeypatch.setenv('AI_INTERNAL_KEY', 'test-internal-key')
    monkeypatch.setattr(live_scan, 'owner', lambda ws: 'test-owner')
    monkeypatch.setattr(websockets.asyncio.client, 'connect', disconnected)
    await live_scan.scan(Socket())


def test_pose_ready_requires_active_pose_registry_and_valid_checkpoint(tmp_path, monkeypatch):
    import hashlib
    import json
    from cardlens_ai import main as ai_main, detection

    checkpoint=tmp_path/"pose.pt";checkpoint.write_bytes(b"weights")
    info={"id":"pose", "checkpoint":str(checkpoint),
          "sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          "validation_passed":True, "task":"pose", "classes":["pokemon_card"],
          "keypoints":["TL","TR","BR","BL"]}
    (tmp_path/"models").mkdir()
    (tmp_path/"models/pose-detectors.json").write_text(json.dumps({"active":"pose","models":[info]}))
    monkeypatch.setattr(detection,"DATA",tmp_path)
    value=ai_main.health()
    assert value["pose_ready"] is True

    checkpoint.write_bytes(b"changed")
    assert ai_main.health()["pose_ready"] is False



def test_pose_toggle_is_opt_in_and_only_accepts_active_registry_id():
    from cardlens_ai.live import configured_pose_model

    assert configured_pose_model({"use_pose": False, "pose_model": "pose-good"}, "pose-good") is None
    assert configured_pose_model({"use_pose": True, "pose_model": "pose-good"}, "pose-good") == "pose-good"
    with pytest.raises(ValueError, match="pose_detector_not_active"):
        configured_pose_model({"use_pose": True, "pose_model": "pose-other"}, "pose-good")


def test_multicard_tilted_visual_consensus_and_180_rotation_ranking():
    from cardlens_ai.live import _recognition_rank
    state = TrackState()
    state.observe(np.zeros(63, dtype=np.uint8), 1)
    visual_reprint = {
        "status": "needs_confirmation",
        "margin": .016,
        "margin_score": "fusion_score",
        "candidates": [
            {"card_id": "base1:54:en", "name": "Metapod", "similarity": .97, "fusion_score": .97, "evidence": {}},
            {"card_id": "lc:54:en", "name": "Metapod", "similarity": .954, "fusion_score": .954, "evidence": {}},
            {"card_id": "base1:10:en", "name": "Caterpie", "similarity": .70, "fusion_score": .70, "evidence": {}},
        ],
        "ocr": {"number": None, "name": "Metapod"},
    }
    for _ in range(3):
        state.accept(visual_reprint)
    assert state.status == "needs_confirmation" and state.votes == 3
    upside_down = {"status": "unknown", "candidates": [{"card_id": "a", "similarity": .77, "fusion_score": .77, "evidence": {}}], "ocr": {"number": None}}
    rotated_180 = {"status": "needs_confirmation", "candidates": [{"card_id": "b", "similarity": .98, "fusion_score": .98, "evidence": {}}], "ocr": {"number": None}}
    assert _recognition_rank(rotated_180) > _recognition_rank(upside_down)
