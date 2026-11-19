from ml.dataset import split_rows

def test_split_keeps_sessions_and_physical_cards_together():
    rows=[{"sample_id":"a","session_id":"s1","physical_card_id":"p1"},{"sample_id":"b","session_id":"s2","physical_card_id":"p1"},{"sample_id":"c","session_id":"s2","physical_card_id":"p2"},{"sample_id":"d","session_id":"s3","physical_card_id":"p2"}]
    result=split_rows(rows)
    assert len({r["split"] for r in result})==1



def test_real_dataset_rejects_split_leakage_and_prepares_labels(tmp_path):
    import json
    import pytest
    from PIL import Image
    from ml.dataset import validate_rows
    from ml.prepare_pose import prepare
    image=tmp_path/"card.jpg";Image.new("RGB",(120,180),"blue").save(image)
    base={"sample_id":"one","session_id":"session-one","physical_card_id":"same-card","card_id":"set:007:en","language":"en","side":"front","variant":"unknown","path":str(image),"source":"own-photo","source_kind":"real_photo","consent":"research_and_evaluation","corners":[[0,0],[119,0],[119,179],[0,179]],"split":"train"}
    other={**base,"sample_id":"two","session_id":"session-two","split":"test"}
    with pytest.raises(ValueError,match="leakage"):validate_rows([base,other])
    with pytest.raises(ValueError,match="consent"):validate_rows([{**base,"consent":"no"}])
    back={**base,"sample_id":"back","side":"back"}
    manifest=tmp_path/"manifest.jsonl";manifest.write_text("\n".join(json.dumps(row) for row in (base,back)),encoding="utf-8")
    counts=prepare(manifest,tmp_path/"pose")
    assert counts=={"train":2,"val":0,"test":0}
    labels=list((tmp_path/"pose/labels/train").glob("*.txt"))
    assert len(labels)==2
    for label in labels:
        values=[float(value) for value in label.read_text().split()]
        assert len(values)==17
        assert values[0]==0
        assert values[5:]==pytest.approx([0,0,2,119/120,0,2,119/120,179/180,2,0,179/180,2])
    config=json.loads((tmp_path/"pose/pose.yaml").read_text(encoding="utf-8"))
    assert config["names"]=={"0":"pokemon_card"}
    assert config["kpt_shape"]==[4,3]
    exported=[json.loads(line) for line in (tmp_path/"pose/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert exported==[base,back]



def test_pose_export_declares_horizontal_corner_mirror_order(tmp_path):
    import json
    from PIL import Image
    from ml.prepare_pose import prepare

    image=tmp_path/"card.jpg";Image.new("RGB",(100,160),"white").save(image)
    row={"sample_id":"s","session_id":"session","card_id":"set:1:en","language":"en","side":"front","variant":"unknown","path":str(image),"source":"owned","source_kind":"real_photo","consent":"research_and_evaluation","corners":[[5,5],[95,5],[95,155],[5,155]],"split":"train"}
    manifest=tmp_path/"manifest.jsonl";manifest.write_text(json.dumps(row))
    prepare(manifest,tmp_path/"pose")
    config=json.loads((tmp_path/"pose/pose.yaml").read_text())
    assert config["flip_idx"]==[1,0,3,2]
