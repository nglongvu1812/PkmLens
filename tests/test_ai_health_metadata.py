import json

from cardlens_ai.main import health


def test_health_reports_active_embedding_kind_without_hardcoding(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR",str(tmp_path))
    root=tmp_path/"models";root.mkdir()
    (root/"registry.json").write_text(json.dumps({"checkpoint_kind":"fine_tuned_projection_frozen_backbone"}))
    assert health()["checkpoint_kind"]=="fine_tuned_projection_frozen_backbone"


def test_health_does_not_claim_pretrained_model_when_registry_is_unavailable(tmp_path,monkeypatch):
    monkeypatch.setenv("DATA_DIR",str(tmp_path))
    assert health()["checkpoint_kind"]=="unavailable"
    assert health()["identification_ready"] is False
