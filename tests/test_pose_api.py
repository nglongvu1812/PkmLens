import hashlib
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from cardlens_ai import training, detection
from cardlens_ai.main import app


@pytest.fixture
def pose_api(tmp_path, monkeypatch):
    data=tmp_path/"datasets/cards";data.mkdir(parents=True)
    (data/"pose.yaml").write_text("{}")
    checkpoint=tmp_path/"models/initializer.pt";checkpoint.parent.mkdir();checkpoint.write_bytes(b"initializer")
    monkeypatch.setattr(training,"DATA",tmp_path)
    monkeypatch.setattr(training,"ROOT",tmp_path)
    monkeypatch.setattr(training,"processes",{})
    monkeypatch.setattr(training,"validate_pose_dataset",lambda root:{"valid":True,"counts":{"train":{"images":2,"objects":2},"val":{"images":1,"objects":1}},"errors":[],"manifest_sha256":"fixture"})
    monkeypatch.setenv("AI_INTERNAL_KEY","test-pose-internal")
    commands=[]
    class Proc:
        def poll(self):return None
    monkeypatch.setattr(training.subprocess,"Popen",lambda cmd,**kwargs:commands.append(cmd) or Proc())
    return tmp_path,commands,TestClient(app),{"x-cardlens-key":"test-pose-internal"}


def test_pose_api_auth_start_status_and_stop(pose_api):
    root,commands,client,headers=pose_api
    body={"dataset":"cards","checkpoint":"initializer.pt","epochs":2,"batch":1,"imgsz":320,"device":"cpu"}
    assert client.post("/v1/training/pose-runs",json=body).status_code==403
    response=client.post("/v1/training/pose-runs",json=body,headers=headers)
    response.raise_for_status();ident=response.json()["id"]
    assert commands[0][:4]==[training.sys.executable,"-m","ml.train_pose","train"]
    assert commands[0][commands[0].index("--checkpoint")+1]==str(root/"models/initializer.pt")
    assert commands[0][commands[0].index("--data")+1]==str(root/"datasets/cards/pose.yaml")
    value=client.get("/v1/training/pose-runs/"+ident,headers=headers).json()
    assert value["task"]=="pose" and value["state"]["status"]=="starting"
    assert value["result"]=={}
    rows=client.get("/v1/training/status",headers=headers).json()["pose_runs"]
    assert any(row["id"]==ident for row in rows)
    response=client.post("/v1/training/pose-runs/"+ident+"/stop",headers=headers)
    response.raise_for_status()
    assert (root/"training"/ident/"STOP").is_file()


@pytest.mark.parametrize("checkpoint",["../outside.pt","C:/private/file.pt","models/initializer.pt","initializer.onnx"])
def test_pose_api_rejects_untrusted_initializer_path(pose_api,checkpoint):
    _,commands,client,headers=pose_api
    response=client.post("/v1/training/pose-runs",json={"dataset":"cards","checkpoint":checkpoint},headers=headers)
    assert response.status_code==422 and commands==[]


def test_pose_resume_uses_optimizer_checkpoint_and_original_configuration(pose_api):
    root,commands,client,headers=pose_api
    run=root/"training/pose-resume";(run/"weights").mkdir(parents=True)
    (run/"weights/last.pt").write_bytes(b"stripped")
    (run/"weights/resume.pt").write_bytes(b"optimizer")
    config={"task":"pose","dataset":str(root/"datasets/cards"),"checkpoint":str(root/"models/initializer.pt"),"epochs":8,"batch":2,"imgsz":480,"scale":"n"}
    (run/"pose-config.json").write_text(json.dumps(config))
    (run/"pose-status.json").write_text(json.dumps({"status":"stopped"}))
    (run/"STOP").touch()
    response=client.post("/v1/training/pose-runs",json={"dataset":"cards","resume_run":"pose-resume","resume_checkpoint":"last.pt","epochs":1,"imgsz":320,"device":"cpu"},headers=headers)
    response.raise_for_status()
    command=commands[0]
    assert "--resume" in command and command[command.index("--checkpoint")+1]==str(run/"weights/resume.pt")
    assert command[command.index("--epochs")+1]=="8" and command[command.index("--imgsz")+1]=="480"
    assert not (run/"STOP").exists()


def test_pose_worker_launch_failure_is_terminal(pose_api,monkeypatch):
    root,_,client,headers=pose_api
    def fail(*args,**kwargs):raise OSError("fixture")
    monkeypatch.setattr(training.subprocess,"Popen",fail)
    response=client.post("/v1/training/pose-runs",json={"dataset":"cards","checkpoint":"initializer.pt"},headers=headers)
    assert response.status_code==503
    status=next((root/"training").glob("*/pose-status.json"))
    assert json.loads(status.read_text())["status"]=="failed"


def test_pose_activation_cannot_trust_boolean_gate_alone(tmp_path,monkeypatch):
    monkeypatch.setattr(training,"DATA",tmp_path);monkeypatch.setattr(training,"processes",{})
    monkeypatch.setattr(training,"unload",lambda:None)
    monkeypatch.setattr(detection,"verify_pose_checkpoint",lambda path:{"task":"pose"},raising=False)
    run=tmp_path/"training/pose-activation";run.mkdir(parents=True)
    weights=run/"best.pt";weights.write_bytes(b"fixture")
    result={"task":"pose","status":"completed","validation_passed":True,"classes":["pokemon_card"],"keypoints":["TL","TR","BR","BL"],"real_photo_validated":True,"checkpoint":str(weights),"sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),"metrics":{"metrics/mAP50(P)":.9,"metrics/precision(P)":.9,"metrics/recall(P)":.9}}
    (run/"pose-result.json").write_text(json.dumps(result));(run/"pose-status.json").write_text(json.dumps({"status":"completed"}))
    with pytest.raises(HTTPException) as failure:training.activate_pose(run.name)
    assert failure.value.status_code==409
    activated=training.activate_pose(run.name,experimental=True)
    assert activated["experimental"] is True
    value=json.loads((tmp_path/"models/pose-detectors.json").read_text())
    assert value["models"][0]["real_photo_validated"] is False
    result["metrics"]["metrics/recall(P)"]=.1
    (run/"pose-result.json").write_text(json.dumps(result))
    with pytest.raises(HTTPException):training.activate_pose(run.name,experimental=True)
