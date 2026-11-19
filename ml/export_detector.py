import argparse
import hashlib
import json
from pathlib import Path


def export(run):
    from ultralytics import YOLO
    import numpy as np
    info=json.loads((run/'result.json').read_text())
    config=json.loads((run/'config.json').read_text())
    model=YOLO(info['checkpoint'])
    result={'status':'exporting','parity_passed':False}
    (run/'export.json').write_text(json.dumps(result))
    try:
        output=Path(model.export(format='onnx',imgsz=config['imgsz'],device='cpu',simplify=False,dynamic=False,opset=17))
        dataset=Path(config['dataset'])
        images=list((dataset/'images/test').glob('*.jpg'))[:10]
        differences=[]
        onnx=YOLO(str(output),task='detect')
        for image in images:
            before=model.predict(str(image),imgsz=config['imgsz'],device='cpu',conf=.45,verbose=False)[0].boxes.xyxy.numpy()
            after=onnx.predict(str(image),imgsz=config['imgsz'],device='cpu',conf=.45,verbose=False)[0].boxes.xyxy.numpy()
            if before.shape!=after.shape:raise ValueError('export_detection_count_differs')
            if len(before):
                before=before[np.argsort(before[:,0])];after=after[np.argsort(after[:,0])]
                differences.append(float(np.max(np.abs(before-after))))
        if not differences:raise ValueError('no_detections_for_parity')
        result={'status':'completed','path':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                'max_box_difference_px':max(differences),'parity_passed':max(differences)<2,'samples':len(images),
                'source_kind':'synthetic_only'}
    except Exception as exc:result={'status':'failed','error':str(exc),'parity_passed':False}
    (run/'export.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);export(p.parse_args().run)
