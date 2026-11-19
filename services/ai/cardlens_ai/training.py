"""Local training jobs with validated paths, real metrics, and audited activation via core."""
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from typing import Literal
import numpy as np
from ml.prepare_pose import validate_pose_dataset
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from .detection import DATA, registry, unload

router = APIRouter(prefix='/v1/training')
log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
job_lock = threading.Lock()
processes = {}


def internal(x_cardlens_key: str = Header(default='')):
    key = os.getenv('AI_INTERNAL_KEY', '')
    if not key or not secrets.compare_digest(x_cardlens_key, key):
        raise HTTPException(403, 'internal_service_only')


def location(kind, ident):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', ident):
        raise HTTPException(422, 'invalid_artifact_id')
    return DATA / kind / ident


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {} if default is None else default


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def validate_dataset(path):
    import yaml
    from PIL import Image
    config = yaml.safe_load((path / 'dataset.yaml').read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('names') not in ({0: 'pokemon_card'}, {'0': 'pokemon_card'}, ['pokemon_card']):
        raise ValueError('dataset_must_have_pokemon_card_class')
    counts, signatures, errors = {}, {}, []
    for split in ('train', 'val', 'test'):
        if split == 'test' and not config.get('test'):
            continue
        image_root = (path / config.get(split, 'images/' + split)).resolve()
        if not image_root.is_relative_to(path.resolve()) or not image_root.is_dir():
            raise ValueError('dataset_paths_must_be_local_split_directories')
        count, objects = 0, 0
        for image in image_root.iterdir():
            if image.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in signatures and signatures[digest] != split:
                errors.append('duplicate_image_across_splits:' + image.name)
            signatures[digest] = split
            with Image.open(image) as im:
                if im.width * im.height > 16_000_000:
                    raise ValueError('dataset_image_too_large')
                im.verify()
            label = path / 'labels' / split / (image.stem + '.txt')
            if not label.is_file():
                errors.append('missing_label:' + image.name)
                continue
            for line in label.read_text().splitlines():
                values = np.array([float(v) for v in line.split()])
                if (len(values) != 5 or not np.isfinite(values).all() or values[0] != 0 or
                        not ((values[1:] >= 0) & (values[1:] <= 1)).all() or min(values[3:]) <= 0 or
                        values[1] - values[3]/2 < -.001 or values[1] + values[3]/2 > 1.001 or
                        values[2] - values[4]/2 < -.001 or values[2] + values[4]/2 > 1.001):
                    errors.append('invalid_label:' + image.name)
                objects += 1
            count += 1
        counts[split] = {'images': count, 'objects': objects}
        if not count or not objects:
            errors.append('empty_split:' + split)
    manifest = path / 'manifest.jsonl'
    groups = {}
    if not manifest.is_file():
        errors.append('manifest_missing')
    else:
        for line in manifest.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            keys = ['source:' + o['group'] for o in row.get('objects', []) if o.get('group')]
            keys += [k + ':' + str(row[k]) for k in ('session_id', 'physical_card_id') if row.get(k)]
            for key in keys:
                if groups.setdefault(key, row['split']) != row['split']:
                    errors.append('source_group_leakage:' + key)
    return {'valid': not errors, 'counts': counts, 'errors': errors[:50],
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None}


@router.get('/status', dependencies=[Depends(internal)])
def status():
    import torch
    gpu = torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info() if gpu else (0, 0)
    datasets = []
    for path in sorted((DATA/'datasets').glob('*/dataset.json')):
        datasets.append({'id': path.parent.name, **read_json(path), 'validation': read_json(path.parent/'validation.json')})
    runs = []
    for path in sorted((DATA/'training').glob('*/config.json'), key=lambda p:p.stat().st_mtime, reverse=True):
        config = read_json(path)
        state = read_json(path.parent/'status.json')
        proc = processes.get(path.parent.name)
        if state.get('status') in ('starting', 'training', 'stopping') and (proc is None or proc.poll() is not None):
            state = {**state, 'status': 'interrupted', 'error': 'worker_not_running_resume_from_checkpoint'}
        history = []
        metrics = path.parent/'results.csv'
        if metrics.exists():
            with metrics.open(encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    try: history.append({k.strip(): float(v) for k,v in row.items()})
                    except (ValueError, TypeError): continue
        runs.append({'id': path.parent.name, 'config': config, 'state': state, 'history': history,
                     'checkpoints': [p.name for p in (path.parent/'weights').glob('*.pt')],
                     'export': read_json(path.parent/'export.json')})
    pose_runs=[]
    for config_path in sorted((DATA/'training').glob('*/pose-config.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        run=config_path.parent
        value=pose_run_status(run.name)
        pose_runs.append(value)
    return {'cuda': gpu, 'device': torch.cuda.get_device_name(0) if gpu else 'CPU',
            'free_vram_mb': free//1024**2, 'total_vram_mb': total//1024**2,
            'datasets': datasets, 'runs': runs, 'pose_runs':pose_runs,'registry': registry()}


class PoseTrainInput(BaseModel):
    dataset: str = Field(min_length=1,max_length=80)
    checkpoint: str | None = Field(default=None,max_length=160)
    scale: Literal['n','s'] = 'n'
    epochs: int = Field(25,ge=1,le=300)
    batch: int = Field(2,ge=1,le=16)
    imgsz: Literal[320,480,640] = 480
    device: Literal['auto','cpu','0'] = 'auto'
    resume_run: str | None = None
    resume_checkpoint: str = 'resume.pt'


def read_pose_history(path):
    history=[]
    if path.is_file():
        try:
            with path.open(encoding='utf-8-sig') as handle:
                for row in csv.DictReader(handle):
                    try:history.append({k.strip():float(v) for k,v in row.items()})
                    except (TypeError,ValueError):continue
        except OSError:pass
    return history[-300:]


def pose_location(kind,ident):
    root=(DATA/kind).resolve()
    path=location(kind,ident).resolve()
    if path==root or not path.is_relative_to(root):
        raise HTTPException(422,'pose_artifact_path_outside_data')
    return path


def pose_run_status(run_id):
    run=pose_location('training',run_id)
    config=read_json(run/'pose-config.json')
    if not config:raise HTTPException(404,'pose_run_not_found')
    state=read_json(run/'pose-status.json')
    proc=processes.get('pose:'+run_id)
    if state.get('status') in ('starting','training','stopping') and (proc is None or proc.poll() is not None):
        # A finished child without a terminal artifact is interrupted, never completed by inference.
        state={**state,'status':'interrupted','error':'worker_not_running_resume_from_checkpoint'}
    return {'id':run_id,'task':'pose','config':config,'state':state,
            'result':read_json(run/'pose-result.json'),
            'checkpoints':[p.name for p in (run/'weights').glob('*.pt')],
            'history':read_pose_history(run/'results.csv')}


def start_pose(body:PoseTrainInput):
    with job_lock:
        if any(proc.poll() is None for proc in processes.values()):
            raise HTTPException(409,'training_already_running')
        dataset=pose_location('datasets',body.dataset)
        try:validation=validate_pose_dataset(dataset)
        except (OSError,ValueError,KeyError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
        if not validation.get('valid'):raise HTTPException(422,validation)
        if body.resume_run:
            run=pose_location('training',body.resume_run)
            config=read_json(run/'pose-config.json')
            if config.get('task')!='pose' or Path(config.get('dataset','')).resolve()!=dataset:
                raise HTTPException(422,'resume_pose_dataset_must_match')
            name=body.resume_checkpoint
            if not re.fullmatch(r'(epoch\d+|last|resume)\.pt',name):raise HTTPException(422,'select_pose_resume_checkpoint')
            if name=='last.pt' and (run/'weights/resume.pt').is_file():name='resume.pt'
            checkpoint=(run/'weights'/name).resolve()
            if not checkpoint.is_relative_to(run) or not checkpoint.is_file():raise HTTPException(404,'pose_resume_checkpoint_not_found')
            if read_json(run/'pose-status.json').get('status')=='completed':
                raise HTTPException(409,'pose_run_completed_start_new_finetune')
            run_id=run.name
            values={key:config.get(key,getattr(body,key)) for key in ('epochs','batch','imgsz','scale')}
            (run/'STOP').unlink(missing_ok=True)
        else:
            if not body.checkpoint or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,150}\.pt',body.checkpoint):
                raise HTTPException(422,'local_pose_initializer_filename_required')
            models_root=(DATA/'models').resolve()
            checkpoint=(models_root/body.checkpoint).resolve()
            if not checkpoint.is_relative_to(models_root) or not checkpoint.is_file():
                raise HTTPException(422,'local_pose_checkpoint_required')
            run_id='pose-'+str(int(time.time()))+'-'+secrets.token_hex(3)
            run=pose_location('training',run_id);run.mkdir(parents=True)
            config=body.model_dump();config.update(dataset=str(dataset),checkpoint=str(checkpoint),task='pose')
            atomic_json(run/'pose-config.json',config)
            values=body.model_dump()
        atomic_json(dataset/'pose-validation.json',validation)
        command=[sys.executable,'-m','ml.train_pose','train','--data',str(dataset/'pose.yaml'),
                 '--checkpoint',str(checkpoint),'--output',str(run),'--epochs',str(values['epochs']),
                 '--batch',str(values['batch']),'--imgsz',str(values['imgsz']),'--scale',values['scale'],'--device',body.device]
        if body.resume_run:command.append('--resume')
        atomic_json(run/'pose-status.json',{'status':'starting','task':'pose'})
        try:
            with (run/'pose-train.log').open('ab') as handle:
                processes['pose:'+run_id]=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1','YOLO_CONFIG_DIR':str(DATA/'yolo-config')})
        except OSError as exc:
            atomic_json(run/'pose-status.json',{'status':'failed','task':'pose','error':'pose_worker_start_failed'})
            raise HTTPException(503,'pose_worker_start_failed') from exc
        return {'id':run_id,'task':'pose','status':'starting'}


@router.post('/pose-datasets/{dataset_id}/validate',dependencies=[Depends(internal)])
def validate_pose(dataset_id:str):
    root=pose_location('datasets',dataset_id)
    try:result=validate_pose_dataset(root)
    except (OSError,ValueError,KeyError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
    atomic_json(root/'pose-validation.json',result)
    return result


@router.get('/pose-runs/{run_id}',dependencies=[Depends(internal)])
def pose_run(run_id:str):
    return pose_run_status(run_id)


@router.post('/pose-runs',dependencies=[Depends(internal)])
def start_pose_route(body:PoseTrainInput):
    return start_pose(body)


@router.post('/pose-runs/{run_id}/stop',dependencies=[Depends(internal)])
def stop_pose(run_id:str):
    run=pose_location('training',run_id)
    if not (run/'pose-config.json').is_file():raise HTTPException(404,'pose_run_not_found')
    state=read_json(run/'pose-status.json')
    if state.get('status') not in ('starting','training','stopping'):
        raise HTTPException(409,'pose_run_not_running')
    (run/'STOP').touch();state.update(status='stopping',task='pose')
    atomic_json(run/'pose-status.json',state)
    return {'id':run_id,'task':'pose','status':'stopping_after_epoch','checkpoint_preserved':True}


@router.post('/datasets/{dataset_id}/validate', dependencies=[Depends(internal)])
def validate(dataset_id: str):
    path = location('datasets', dataset_id)
    try:
        result = validate_dataset(path)
        atomic_json(path/'validation.json', result)
        return result
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


class TrainInput(BaseModel):
    dataset: str
    model: Literal['n','s','m'] = 'n'
    epochs: int = Field(25, ge=1, le=300)
    batch: int = Field(4, ge=1, le=16)
    imgsz: Literal[320,480,640] = 480
    device: Literal['auto','cpu','0'] = 'auto'
    resume_run: str | None = None
    checkpoint: str | None = None


@router.post('/runs', dependencies=[Depends(internal)])
def start(body: TrainInput):
    with job_lock:
        if any(p.poll() is None for p in processes.values()):
            raise HTTPException(409, 'training_already_running')
        dataset = location('datasets', body.dataset)
        try: validation = validate_dataset(dataset)
        except (OSError, ValueError, KeyError) as exc: raise HTTPException(422, str(exc)) from exc
        if not validation['valid']:
            raise HTTPException(422, validation)
        atomic_json(dataset/'validation.json', validation)
        ident = 'yolo26'+body.model+'-'+str(int(time.time()))+'-'+secrets.token_hex(3)
        output = location('training', ident)
        output.mkdir(parents=True)
        command = [sys.executable, '-m', 'ml.train_detection', str(dataset), str(output), '--model', body.model,
                   '--epochs', str(body.epochs), '--batch', str(body.batch), '--imgsz', str(body.imgsz), '--device', body.device]
        if body.resume_run:
            previous = location('training', body.resume_run)
            if not body.checkpoint or not re.fullmatch(r'(epoch\d+|last|resume)\.pt', body.checkpoint):
                raise HTTPException(422, 'select_resumable_checkpoint')
            checkpoint = previous/'weights'/body.checkpoint
            if body.checkpoint == 'last.pt' and (previous/'weights/resume.pt').is_file():
                checkpoint = previous/'weights/resume.pt'
            if not checkpoint.is_file():raise HTTPException(404,'checkpoint_not_found')
            # Ultralytics resumes into its original run directory.
            output.rmdir();output=previous;ident=previous.name
            (output/'STOP').unlink(missing_ok=True)
            command[4] = str(output)
            command.extend(['--resume',str(checkpoint)])
        if not body.resume_run:
            atomic_json(output/'config.json', body.model_dump())
        atomic_json(output/'status.json', {'status':'starting'})
        with (output/'train.log').open('ab') as handle:
            processes[ident]=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONIOENCODING':'utf-8','YOLO_CONFIG_DIR':str(DATA/'yolo-config')})
        log.info('training_started id=%s',ident)
        return {'id':ident,'status':'starting'}


@router.post('/runs/{run_id}/stop', dependencies=[Depends(internal)])
def stop(run_id:str):
    path=location('training',run_id)
    if not path.exists():raise HTTPException(404)
    (path/'STOP').touch()
    return {'id':run_id,'status':'stopping_after_epoch','checkpoint_preserved':True}


class Activate(BaseModel):
    experimental: bool = False


@router.post('/runs/{run_id}/activate', dependencies=[Depends(internal)])
def activate(run_id:str,body:Activate):
    path=location('training',run_id)
    result=read_json(path/'result.json')
    if not result.get('validation_passed') or result.get('status')!='completed':
        raise HTTPException(409,'validation_gate_not_passed')
    if not result.get('real_photo_validated') and not body.experimental:
        raise HTTPException(409,'synthetic_checkpoint_requires_experimental_mode')
    checkpoint=Path(result['checkpoint']).resolve()
    if not checkpoint.is_relative_to(path) or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=result['sha256']:
        raise HTTPException(409,'checkpoint_integrity_failed')
    data=registry()
    entry={**result,'id':run_id}
    data['models']=[m for m in data['models'] if m['id']!=run_id]+[entry]
    data['active']=run_id
    atomic_json(DATA/'models/detectors.json',data)
    unload()
    log.info('detector_activated id=%s experimental=%s',run_id,body.experimental)
    return {'active':run_id,'experimental':not result.get('real_photo_validated')}


def activate_pose(run_id,experimental=False):
    from .detection import verify_pose_checkpoint
    import math
    with job_lock:
        if any(proc.poll() is None for proc in processes.values()):raise HTTPException(409,'training_busy')
        run=pose_location('training',run_id)
        result=read_json(run/'pose-result.json')
        state=read_json(run/'pose-status.json')
        if result.get('task')!='pose' or result.get('status')!='completed' or state.get('status')!='completed' or not result.get('validation_passed'):
            raise HTTPException(409,'pose_validation_gate_not_passed')
        metrics=result.get('metrics',{})
        checks=(('metrics/mAP50(P)','POSE_MIN_MAP50',.75),
                ('metrics/precision(P)','POSE_MIN_PRECISION',.70),
                ('metrics/recall(P)','POSE_MIN_RECALL',.70))
        for metric,key,default in checks:
            try:value=float(metrics[metric]);minimum=float(os.getenv(key,str(default)))
            except (ValueError,TypeError,KeyError):raise HTTPException(409,'pose_validation_metrics_missing')
            if not math.isfinite(value) or not math.isfinite(minimum) or not 0<=minimum<=1 or not minimum<=value<=1:
                raise HTTPException(409,'pose_validation_metrics_below_gate')
        if not experimental:
            # Camera accuracy requires independent real-photo test evidence, not a result flag.
            raise HTTPException(409,'pose_checkpoint_requires_experimental_mode')
        if result.get('classes')!=['pokemon_card'] or result.get('keypoints')!=['TL','TR','BR','BL']:
            raise HTTPException(409,'pose_schema_invalid')
        checkpoint=Path(result.get('checkpoint','')).resolve()
        if not checkpoint.is_relative_to(run) or not checkpoint.is_file():raise HTTPException(409,'pose_checkpoint_integrity_failed')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=result.get('sha256'):raise HTTPException(409,'pose_checkpoint_integrity_failed')
        try:verify_pose_checkpoint(checkpoint)
        except (OSError,ValueError,RuntimeError) as exc:raise HTTPException(409,'pose_checkpoint_schema_failed') from exc
        registry_path=DATA/'models/pose-detectors.json'
        data=read_json(registry_path,{'active':None,'models':[]})
        entry={**result,'id':run_id,'checkpoint':str(checkpoint),'real_photo_validated':False}
        data['models']=[model for model in data.get('models',[]) if model.get('id')!=run_id]+[entry]
        data['active']=run_id
        atomic_json(registry_path,data);unload()
        log.info('pose_detector_activated id=%s experimental=true',run_id)
        return {'active':run_id,'experimental':True}


@router.post('/pose-runs/{run_id}/activate',dependencies=[Depends(internal)])
def activate_pose_route(run_id:str,body:Activate):
    return activate_pose(run_id,experimental=body.experimental)


@router.post('/runs/{run_id}/export', dependencies=[Depends(internal)])
def export(run_id:str):
    path=location('training',run_id)
    result=read_json(path/'result.json')
    if not result.get('validation_passed'):raise HTTPException(409,'validation_gate_not_passed')
    with job_lock:
        if any(p.poll() is None for p in processes.values()):raise HTTPException(409,'training_busy')
        with (path/'export.log').open('ab') as handle:
            processes[run_id+'-export']=subprocess.Popen([sys.executable,'-m','ml.export_detector',str(path)],
                cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    return {'status':'exporting'}
