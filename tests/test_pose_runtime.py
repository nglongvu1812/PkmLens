import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from cardlens_ai import detection
from cardlens_ai.imaging import warp
from test_pose_training import dataset as pose_dataset, runtime as pose_training_runtime


def test_pose_registry_requires_checksum_validated_corner_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pose.pt"
    checkpoint.write_bytes(b"pose weights")
    info = {"id": "pose-1", "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "validation_passed": True, "real_photo_validated": False,
            "task": "pose", "classes": ["pokemon_card"], "keypoints": ["TL", "TR", "BR", "BL"]}
    monkeypatch.setattr(detection, "DATA", tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models/pose-detectors.json").write_text(json.dumps({"active": "pose-1", "models": [info]}))

    resolved, path = detection.resolve_pose_model()

    assert resolved["id"] == "pose-1" and path == checkpoint
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        detection.resolve_pose_model()


def test_pose_corners_are_ordered_convex_and_warp_to_card_shape():
    image = Image.new("RGB", (200, 200), "white")
    corners = [[30, 20], [170, 35], [155, 180], [20, 160]]

    warped, ordered = detection.rectify_corners(image, corners)

    assert warped.size == (504, 704)
    assert ordered == corners
    with pytest.raises(ValueError, match="corner_order"):
        warp(image, [[30, 20], [20, 160], [155, 180], [170, 35]])
    with pytest.raises(ValueError, match="corners_not_convex"):
        detection.rectify_corners(image, [[20, 20], [180, 20], [40, 40], [20, 180]])


def test_pose_runtime_uses_real_keypoints_and_returns_boxes(tmp_path, monkeypatch):
    import sys
    import types
    import torch

    checkpoint = tmp_path / "pose.pt"
    checkpoint.write_bytes(b"weights")
    info = {"id": "pose", "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "validation_passed": True, "task": "pose", "classes": ["pokemon_card"],
            "keypoints": ["TL", "TR", "BR", "BL"]}
    monkeypatch.setattr(detection, "DATA", tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models/pose-detectors.json").write_text(json.dumps({"active": "pose", "models": [info]}))

    class Boxes:
        xyxy = torch.tensor([[10, 20, 190, 180]], dtype=torch.float32)
        conf = torch.tensor([0.9], dtype=torch.float32)
        cls = torch.tensor([0], dtype=torch.float32)
        def __len__(self): return 1
        def cpu(self): return self
        def numpy(self): return np.column_stack((self.xyxy.numpy(), self.conf.numpy(), self.cls.numpy()))

    class Keypoints:
        xy = torch.tensor([[[20, 20], [180, 20], [180, 180], [20, 180]]], dtype=torch.float32)
        conf = torch.full((1, 4), 0.95, dtype=torch.float32)

    class Result:
        boxes = Boxes()
        keypoints = Keypoints()

    class Model:
        task = "pose"
        names = {0: "pokemon_card"}
        kpt_shape = (4, 3)
        def predict(self, *args, **kwargs): return [Result()]

    class YOLO:
        def __new__(cls, path, task=None): return Model()

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=YOLO))
    monkeypatch.setenv("DETECTOR_DEVICE", "cpu")

    boxes, corners, model_info = detection.detect_pose(Image.new("RGB", (200, 200)), size=320)

    assert boxes.shape == (1, 6) and boxes[0, 4] == pytest.approx(0.9)
    assert corners.tolist() == [[[20, 20], [180, 20], [180, 180], [20, 180]]]
    assert model_info["id"] == "pose"


def test_pose_training_entry_uses_pose_yaml_and_saves_real_pose_metrics(tmp_path, pose_dataset, pose_training_runtime):
    from ml import train_pose
    result=train_pose.train(pose_dataset/'pose.yaml',pose_training_runtime['checkpoint'],tmp_path/'run',
                            epochs=2,batch=1,imgsz=320,device='cpu')
    call=pose_training_runtime['train'][0]
    assert call['data']==str(pose_dataset/'pose.yaml')
    assert call['fliplr']==0 and call['flipud']==0 and call['mosaic']==0
    assert result['validation_passed'] and result['task']=='pose'
    assert result['metrics']['metrics/mAP50(P)']==.88
    assert (tmp_path/'run/pose-result.json').is_file()

def test_pose_activation_requires_passed_validation_and_contained_checkpoint(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from cardlens_ai import training

    monkeypatch.setattr(training,"DATA",tmp_path)
    monkeypatch.setattr(training,"processes",{})
    monkeypatch.setattr(detection,"verify_pose_checkpoint",lambda path:{"task":"pose"},raising=False)
    run=tmp_path/"training/pose-run";run.mkdir(parents=True)
    checkpoint=run/"best.pt";checkpoint.write_bytes(b"trained pose weights")
    result={"status":"completed","task":"pose","classes":["pokemon_card"],
            "keypoints":["TL","TR","BR","BL"],"validation_passed":True,
            "checkpoint":str(checkpoint),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "real_photo_validated":False,"metrics":{"metrics/mAP50(P)":.9,"metrics/precision(P)":.9,"metrics/recall(P)":.9}}
    (run/"pose-status.json").write_text(json.dumps({"status":"completed"}))
    (run/"pose-result.json").write_text(json.dumps(result))

    activated=training.activate_pose("pose-run",experimental=True)

    assert activated["active"]=="pose-run" and activated["experimental"] is True
    registry=json.loads((tmp_path/"models/pose-detectors.json").read_text())
    assert registry["models"][0]["keypoints"]==["TL","TR","BR","BL"]

    result["validation_passed"]=False
    (run/"pose-result.json").write_text(json.dumps(result))
    with pytest.raises(HTTPException) as error:
        training.activate_pose("pose-run",experimental=True)
    assert error.value.status_code==409


def test_pose_quad_is_projected_as_normalized_source_coordinates():
    from cardlens_ai.detection import normalize_corners

    result=normalize_corners([[20,10],[180,10],[180,90],[20,90]],200,100)

    assert np.allclose(result, [[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]])



def test_pose_dataset_validator_accepts_four_corner_labels_and_rejects_bad_rows(tmp_path):
    from ml.prepare_pose import validate_pose_dataset

    root=tmp_path/"pose"
    for split in ("train","val"):
        image_dir=root/"images"/split; label_dir=root/"labels"/split
        image_dir.mkdir(parents=True);label_dir.mkdir(parents=True)
        sample_id="sample-"+split;name=hashlib.sha256(sample_id.encode()).hexdigest()
        Image.new("RGB",(100,100),"white" if split=="train" else "blue").save(image_dir/(name+".jpg"))
        (label_dir/(name+".txt")).write_text("0 .5 .5 .8 .8 .1 .1 2 .9 .1 2 .9 .9 2 .1 .9 2\n")
        if split=="train":
            train_row={"sample_id":sample_id,"split":split,"source_kind":"real_photo","source":"own-photo","consent":"research_and_evaluation","session_id":"session-train","physical_card_id":"physical-train","objects":[{"group":"art-train","corners":[[10,10],[90,10],[90,90],[10,90]]}]}
        else:
            val_row={"sample_id":sample_id,"split":split,"source_kind":"real_photo","source":"own-photo","consent":"research_and_evaluation","session_id":"session-val","physical_card_id":"physical-val","objects":[{"group":"art-val","corners":[[10,10],[90,10],[90,90],[10,90]]}]}
    (root/"manifest.jsonl").write_text(json.dumps(train_row)+"\n"+json.dumps(val_row)+"\n")
    config={"path":str(root),"train":"images/train","val":"images/val","test":None,"kpt_shape":[4,3],"flip_idx":[1,0,3,2],"names":{"0":"pokemon_card"}}
    (root/"pose.yaml").write_text(json.dumps(config))

    assert validate_pose_dataset(root)["valid"]
    train_label=next((root/"labels/train").glob("*.txt"))
    train_label.write_text("0 .5 .5 .8 .8 .1 .1 2 .9 .1 2 .9 .9 2\n")
    assert any(error.startswith("invalid_pose_label:") for error in validate_pose_dataset(root)["errors"])


def test_pose_warm_start_builds_custom_head_then_transfers_local_detector_weights(tmp_path, pose_dataset, pose_training_runtime):
    from ml import train_pose
    result=train_pose.train(pose_dataset/'pose.yaml',pose_training_runtime['checkpoint'],tmp_path/'run',
                            epochs=1,batch=1,imgsz=320,device='cpu')
    assert pose_training_runtime['instances'][0].path.name=='yolo26n-card-pose.yaml'
    assert pose_training_runtime['initializer']==str(pose_training_runtime['checkpoint'])
    assert result['classes']==['pokemon_card'] and result['keypoints']==['TL','TR','BR','BL']

def test_pose_train_api_validates_dataset_and_confines_local_checkpoint(tmp_path, monkeypatch):
    from cardlens_ai import training

    data=tmp_path/"datasets/pose";data.mkdir(parents=True)
    (data/"pose.yaml").write_text("pose")
    checkpoint=tmp_path/"models/init.pt";checkpoint.parent.mkdir();checkpoint.write_bytes(b"init")
    commands=[]
    class Proc:
        def poll(self):return None
    monkeypatch.setattr(training,"DATA",tmp_path)
    monkeypatch.setattr(training,"ROOT",tmp_path)
    monkeypatch.setattr(training,"processes",{})
    monkeypatch.setattr(training,"validate_pose_dataset",lambda path:{"valid":True,"errors":[],"counts":{"train":{"images":1,"objects":1},"val":{"images":1,"objects":1}}})
    monkeypatch.setattr(training.subprocess,"Popen",lambda command,**kwargs:commands.append(command) or Proc())
    body=training.PoseTrainInput(dataset="pose",checkpoint="init.pt",epochs=2,batch=1,imgsz=320,device="cpu")

    result=training.start_pose(body)

    assert result["task"]=="pose" and result["status"]=="starting"
    assert commands[0][2:4]==["ml.train_pose","train"]
    assert commands[0][commands[0].index("--checkpoint")+1]==str(checkpoint)
    assert commands[0][commands[0].index("--output")+1]==str(tmp_path/"training"/result["id"])


def test_pose_train_api_rejects_checkpoint_path_outside_data_root(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from cardlens_ai import training
    data=tmp_path/"datasets/pose";data.mkdir(parents=True);(data/"pose.yaml").write_text("pose")
    monkeypatch.setattr(training,"DATA",tmp_path)
    monkeypatch.setattr(training,"validate_pose_dataset",lambda path:{"valid":True,"errors":[]})
    with pytest.raises(HTTPException) as error:
        training.start_pose(training.PoseTrainInput(dataset="pose",checkpoint="../outside.pt",device="cpu"))
    assert error.value.status_code==422


def test_pose_detections_are_wrapped_in_ultralytics_boxes_for_tracker():
    from cardlens_ai.detection import tracker_input

    raw=np.array([[10,20,90,180,.9,0]],dtype=np.float32)
    boxes=tracker_input(raw,(200,100))

    assert boxes.xyxy.tolist()==[[10,20,90,180]]
    assert boxes.conf.tolist()==pytest.approx([.9])
    assert boxes.cls.tolist()==[0]



def test_synthetic_scene_export_keeps_pixels_corners_and_frozen_splits(tmp_path):
    from ml.pose_dataset import prepare_scenes
    from ml.prepare_pose import validate_pose_dataset

    source=tmp_path/"source"; rows=[]
    for split in ("train","val","test"):
        image_dir=source/"images"/split; label_dir=source/"labels"/split
        image_dir.mkdir(parents=True);label_dir.mkdir(parents=True)
        sample=split+"-00001"; image=Image.new("RGB",(100,160),(30,80,40 if split=="train" else 120 if split=="val" else 220))
        image.save(image_dir/(sample+".jpg"))
        corners=[[10,10],[90,10],[90,150],[10,150]]
        bbox="0 .5 .5 .8 .875"
        (label_dir/(sample+".txt")).write_text(bbox)
        rows.append({"sample_id":sample,"split":split,"source_kind":"synthetic_catalog_scene",
                     "sha256":hashlib.sha256((image_dir/(sample+".jpg")).read_bytes()).hexdigest(),
                     "objects":[{"group":"art-"+split,"corners":corners,"source":"card-"+split+".webp"}]})
    (source/"manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    output=tmp_path/"pose-output"

    summary=prepare_scenes(source,output)

    assert summary["counts"]=={"train":1,"val":1,"test":1}
    label=next((output/"labels/train").glob("*.txt"))
    values=[float(x) for x in label.read_text().split()]
    assert len(values)==17 and values[0]==0
    assert values[5:]==pytest.approx([.1,.0625,2,.9,.0625,2,.9,.9375,2,.1,.9375,2])
    config=json.loads((output/"pose.yaml").read_text())
    assert config["kpt_shape"]==[4,3] and config["flip_idx"]==[1,0,3,2]
    copied=next((output/"images/train").iterdir())
    assert copied.read_bytes()==(source/"images/train/train-00001.jpg").read_bytes()
    assert validate_pose_dataset(output)["valid"]


def test_synthetic_scene_export_rejects_inconsistent_box_and_quad(tmp_path):
    from ml.pose_dataset import prepare_scenes

    source=tmp_path/"source"
    for split in ("train","val","test"):
        image_dir=source/"images"/split;label_dir=source/"labels"/split
        image_dir.mkdir(parents=True);label_dir.mkdir(parents=True)
        sample=split+"-1";image=Image.new("RGB",(100,100),"white");image.save(image_dir/(sample+".jpg"))
        (label_dir/(sample+".txt")).write_text("0 .5 .5 .4 .4")
    rows=[]
    for split in ("train","val","test"):
        image=source/"images"/split/(split+"-1.jpg")
        rows.append({"sample_id":split+"-1","split":split,"sha256":hashlib.sha256(image.read_bytes()).hexdigest(),"objects":[{"group":"group-"+split,"corners":[[10,10],[90,10],[90,90],[10,90]]}]})
    (source/"manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match="box_does_not_match"):
        prepare_scenes(source,tmp_path/"out")


def test_custom_pose_model_yaml_overrides_keypoints_before_weight_transfer(tmp_path):
    from ml.train_pose import create_pose_model_yaml
    from ultralytics import YOLO

    path=create_pose_model_yaml(tmp_path/"yolo26n-card-pose.yaml","n")
    model=YOLO(str(path),task="pose")

    assert model.model.kpt_shape==[4,3]
    assert model.model.yaml["nc"]==1
    assert path.is_file()


def test_custom_pose_yaml_transfers_local_card_detector_weights(tmp_path, pose_dataset, pose_training_runtime):
    from ml import train_pose
    result=train_pose.train(pose_dataset/'pose.yaml',pose_training_runtime['checkpoint'],tmp_path/'run',
                            epochs=1,batch=1,imgsz=320,device='cpu',scale='s')
    assert pose_training_runtime['instances'][0].path.name=='yolo26s-card-pose.yaml'
    assert result['initial_checkpoint']==str(pose_training_runtime['checkpoint'])
    assert not (tmp_path/'pose-warm-start.pt').exists()

