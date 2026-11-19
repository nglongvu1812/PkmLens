"""Deterministic synthetic data and leakage-safe labeled-manifest splitting."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image, ImageFilter

def split_rows(rows):
    groups={}
    def root(x):
        groups.setdefault(x,x)
        if groups[x]!=x:
            groups[x]=root(groups[x])
        return groups[x]
    for row in rows:
        session="session:"+row["session_id"]
        if row.get("physical_card_id"):
            physical="physical:"+row["physical_card_id"]
            groups[root(session)]=root(physical)
    assigned={}
    for row in rows:
        group=root("session:"+row["session_id"])
        bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%10
        assigned[row["sample_id"]]="test" if bucket==0 else "val" if bucket==1 else "train"
    return [{**r,"split":assigned[r["sample_id"]]} for r in rows]

def generate(count=30,seed=42):
    rng=random.Random(seed)
    gallery=sorted(Path("data/gallery").glob("*.webp"))
    if not gallery:
        raise ValueError("Import gallery first")
    output=Path("data/synthetic")/str(seed)
    output.mkdir(parents=True,exist_ok=True)
    (output/"images").mkdir(exist_ok=True)
    (output/"labels").mkdir(exist_ok=True)
    rows=[]
    for i in range(count):
        path=gallery[i%len(gallery)]
        card=Image.open(path).convert("RGB").resize((240,336))
        canvas=np.full((640,640,3),rng.randint(70,180),dtype=np.uint8)
        center=np.array([320,320])
        angle=rng.uniform(-180,180)*np.pi/180
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        quad=np.array([[-120,-168],[120,-168],[120,168],[-120,168]])@rot.T+center
        quad+=np.array([[rng.uniform(-15,15),rng.uniform(-15,15)] for _ in range(4)])
        matrix=cv2.getPerspectiveTransform(np.float32([[0,0],[239,0],[239,335],[0,335]]),quad.astype("float32"))
        warped=cv2.warpPerspective(np.array(card),matrix,(640,640))
        mask=cv2.warpPerspective(np.full((336,240),255,dtype=np.uint8),matrix,(640,640))
        canvas[mask>0]=warped[mask>0]
        image=Image.fromarray(canvas).filter(ImageFilter.GaussianBlur(rng.uniform(0,.8)))
        sample=f"synthetic-{seed}-{i:06}"
        target=output/"images"/(sample+".jpg")
        image.save(target,quality=rng.randint(75,95))
        x,y=quad.min(axis=0)/640
        right,bottom=quad.max(axis=0)/640
        values=[0,(x+right)/2,(y+bottom)/2,right-x,bottom-y]
        for px,py in quad/640:
            values.extend([px,py,2])
        (output/"labels"/(sample+".txt")).write_text(" ".join(map(str,values)),encoding="utf-8")
        set_id,number=path.stem.rsplit("-",1)
        rows.append({"sample_id":sample,"session_id":f"synthetic-{seed}-{path.stem}","physical_card_id":None,"card_id":set_id+":"+number+":en","language":"en","variant":"unknown","side":"front","source_kind":"synthetic_catalog_transform","path":str(target),"corners":quad.tolist(),"mode":"manual_corners","seed":seed,"sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"source":str(path),"consent":"catalog_research_only"})
    rows=split_rows(rows)
    (output/"manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    print(output/"manifest.jsonl")
    return rows

def validate_rows(rows):
    ids=set()
    sessions={}
    physical={}
    for row in rows:
        for field in ("sample_id","session_id","card_id","language","side","variant","source","consent","path"):
            if field not in row:raise ValueError("Missing field: "+field)
        if not row["session_id"] or row["sample_id"] in ids:raise ValueError("Missing session or duplicate sample_id")
        ids.add(row["sample_id"])
        if row["side"] not in ("front","back"):raise ValueError("Unknown card side")
        if row.get("source_kind")=="real_photo" and row["consent"] not in ("opt_in_training","research_and_evaluation"):
            raise ValueError("Real-photo usage requires explicit scoped consent")
        image_path=Path(row["path"])
        if not image_path.is_file():raise ValueError("Missing image")
        from cardlens_ai.imaging import decode,warp
        image=decode(image_path.read_bytes())
        if row.get("sha256") and hashlib.sha256(image_path.read_bytes()).hexdigest()!=row["sha256"]:raise ValueError("Image checksum mismatch")
        if row.get("corners") is not None:warp(image,row["corners"])
        if row.get("split"):
            if row["split"] not in ("train","val","test"):raise ValueError("Invalid split")
            for group,value in ((sessions,row["session_id"]),(physical,row.get("physical_card_id"))):
                if value and value in group and group[value]!=row["split"]:raise ValueError("Session/physical-card split leakage")
                if value:group[value]=row["split"]
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["generate","split","validate"])
    p.add_argument("--manifest",type=Path)
    p.add_argument("--count",type=int,default=30)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.action=="generate":
        generate(a.count,a.seed)
    else:
        rows=[json.loads(line) for line in a.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        validate_rows(rows)
        result=split_rows(rows)
        if a.action=="split":
            target=a.manifest.with_name(a.manifest.stem+"-split.jsonl")
            target.write_text("\n".join(json.dumps(r) for r in result),encoding="utf-8")
            print(target)
        else:
            print("Valid records:",len(rows))

