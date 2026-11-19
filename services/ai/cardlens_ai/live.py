"""Latest-frame WS inference with independent bounded recognition work."""
import asyncio
import json
import logging
import math
import os
import secrets
import time
from dataclasses import dataclass, field
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import ImageOps
import numpy as np
from .imaging import decode, phash
from .identity import _name, effective_margin
from .detection import (detect, detect_pose, new_tracker, rectify, rectify_corners, quality,
                        registry, verified_pose_metadata, normalize_corners, tracker_input,
                        validate_pose_corners, unload)
from .recognition_config import DEFAULT_POLICY

router=APIRouter()
log=logging.getLogger(__name__)
MAX_FRAME=1_500_000
active_sessions=0
recognition_gate=asyncio.Semaphore(1)


def put_latest(queue, value):
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(value)


def configured_pose_model(config, active):
    model_id=config.get('pose_model') or None
    if config.get('use_pose') and (not model_id or model_id!=active):
        raise ValueError('pose_detector_not_active')
    return model_id if config.get('use_pose') else None


def validated_settings(config):
    """Validate client values before changing the accepted session configuration."""
    request_id=config.get('configuration_id')
    if request_id is not None and (type(request_id) is not int or not 0<=request_id<=2**53-1):
        raise ValueError('invalid_config')
    try:
        conf=float(config.get('confidence',.45))
        size=float(config.get('size',480))
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('invalid_config') from exc
    if (isinstance(config.get('confidence'),bool) or not math.isfinite(conf) or not .1<=conf<=.95
            or isinstance(config.get('size'),bool) or size not in (320,480,640)
            or type(config.get('use_pose',False)) is not bool):
        raise ValueError('invalid_config')
    for name in ('model','pose_model'):
        value=config.get(name)
        if value is not None and (not isinstance(value,str) or len(value)>128):
            raise ValueError('invalid_config')
    return dict(confidence=conf,size=int(size),model=config.get('model') or None,
                pose_model=config.get('pose_model') or None),request_id


@dataclass
class BestFrame:
    image: object
    signature: object
    sharpness: float
    captured_at: float


@dataclass
class TrackState:
    generation: str = field(default_factory=lambda:secrets.token_hex(6))
    signature: object = None
    identity_signature: object = None
    candidates: list = field(default_factory=list)
    last_card: str | None = None
    votes: int = 0
    status: str = "reading"
    last_recognized: float = 0
    seen: float = 0
    visible_at: float = 0
    best_quality: float = 0
    pending: bool = False
    usable: bool = False
    ocr: dict = field(default_factory=dict)
    best_frame: BestFrame | None = None
    smoothed_corners: object = None
    corners_seen: float = 0

    def reset(self):
        self.generation=secrets.token_hex(6)
        self.signature=None; self.identity_signature=None
        self.candidates=[]; self.votes=0; self.last_card=None; self.status="reading"
        self.last_recognized=0; self.best_quality=0; self.ocr={}
        self.best_frame=None; self.pending=False; self.usable=False
        self.smoothed_corners=None; self.corners_seen=0

    def observe(self, signature, now, policy=DEFAULT_POLICY):
        anchor=self.identity_signature if self.identity_signature is not None else self.signature
        changed=anchor is not None and (now-self.seen>policy.signature_timeout_seconds
            or (signature is not None and (np.shape(signature)!=np.shape(anchor)
                or np.mean(signature != anchor)>policy.card_change_ratio)))
        if changed:
            self.reset()
        # A generation has a fixed anchor; rolling comparisons hide gradual card changes.
        if self.signature is None and signature is not None:
            self.signature=np.array(signature,copy=True)
        self.seen=now
        return changed

    def should_recognize(self, now, sharpness, policy=DEFAULT_POLICY):
        if self.votes >= policy.consensus_frames:
            return False
        return (now-self.last_recognized>policy.recognition_interval_seconds
                or sharpness>self.best_quality*policy.best_frame_improvement_ratio
                or now-self.last_recognized>policy.unconfirmed_refresh_seconds)

    def offer_frame(self, image, signature, sharpness, now, policy=DEFAULT_POLICY):
        if self.votes >= policy.consensus_frames:
            return
        previous=self.best_frame
        if (previous is None or now-previous.captured_at>policy.best_frame_max_age_seconds
                or sharpness>=previous.sharpness):
            self.best_frame=BestFrame(image,np.array(signature,copy=True),sharpness,now)

    def take_best_frame(self, now, policy=DEFAULT_POLICY):
        best=self.best_frame
        self.best_frame=None
        return best if best is not None and now-best.captured_at<=policy.best_frame_max_age_seconds else None

    def smooth_corners(self, corners, width, height, now, policy=DEFAULT_POLICY):
        points=validate_pose_corners(corners,width,height,policy)
        if self.smoothed_corners is not None and now-self.corners_seen<=policy.signature_timeout_seconds:
            jump=np.max(np.linalg.norm(points-self.smoothed_corners,axis=1))/np.hypot(width,height)
            if jump>policy.pose_max_corner_jump_ratio:
                raise ValueError("pose_corner_jump")
            alpha=policy.pose_corner_smoothing_alpha
            points=validate_pose_corners(alpha*points+(1-alpha)*self.smoothed_corners,width,height,policy)
        self.smoothed_corners=points.copy(); self.corners_seen=now
        return points

    def accept(self, result, policy=DEFAULT_POLICY, signature=None):
        candidates=result["candidates"]
        previous_name=_name(self.candidates[0].get('name')) if self.candidates else ''
        previous_ocr=dict(self.ocr)
        self.candidates=candidates; self.ocr={k:v for k,v in result["ocr"].items() if k!='crop'}
        if not candidates:
            self.status="unknown"; self.votes=0; self.last_card=None; self.identity_signature=None; return
        best=candidates[0]
        margin=best['similarity']-(candidates[1]['similarity'] if len(candidates)>1 else 0)
        if result.get('margin_score')=='fusion_score':margin=result.get('margin')
        check_margin=(effective_margin(candidates)
                      if len(candidates)>1 and isinstance(margin,(int,float)) and math.isfinite(margin) and margin<policy.min_margin
                      else margin)
        evidence=best.get('evidence',{})
        has_evidence=(evidence.get('number_match') is True
                      or (evidence.get('number_match') is None and self.ocr.get('number') is None
                          and (evidence.get('name_match') is True or best['similarity']>=max(policy.min_similarity,.68))))
        eligible=(result.get('status','needs_confirmation')=='needs_confirmation'
                  and best['similarity']>=policy.min_similarity
                  and isinstance(margin,(int,float)) and math.isfinite(margin)
                  and isinstance(check_margin,(int,float)) and math.isfinite(check_margin) and check_margin>=policy.min_margin
                  and has_evidence
                  and all(evidence.get(key) is not False for key in
                          ('number_match','total_match','name_match','set_code_match','language_match')))
        same_card=best['card_id']==self.last_card or (self.ocr.get('number') is None and bool(previous_name) and _name(best.get('name'))==previous_name)
        if eligible and same_card and self.ocr.get('number') is None and previous_ocr.get('number') is not None:
            self.ocr['number']=previous_ocr['number']
            if self.ocr.get('total') is None and previous_ocr.get('total') is not None:
                self.ocr['total']=previous_ocr['total']
        self.votes=self.votes+1 if eligible and same_card else 1 if eligible else 0
        self.last_card=best['card_id'] if eligible else None
        self.status='needs_confirmation' if self.votes>=policy.consensus_frames else 'unknown' if not eligible else 'stabilizing'
        if self.votes>=policy.consensus_frames:
            anchor=signature if signature is not None else self.signature
            self.identity_signature=np.array(anchor,copy=True) if anchor is not None else None
            self.best_frame=None
        else:self.identity_signature=None


def prepare_crop(image, box, corners, policy=DEFAULT_POLICY):
    if corners is not None:
        flat,ordered=rectify_corners(image,corners,policy)
        correction='yolo26_pose'
    else:
        flat,correction=rectify(image,box)
        ordered=None
    q=quality(flat,policy)
    return flat,ordered,q,phash(flat) if q['usable'] else None,correction


def decode_frame(raw):
    image=decode(raw)
    if image.width*image.height>1920*1080:
        raise ValueError('frame_resolution_limit')
    return image


def track_frame(tracker, boxes, image):
    return tracker.update(tracker_input(boxes,(image.height,image.width)),np.asarray(image)[:,:,::-1])


async def run_thread(function, *args):
    """Cancellation cannot release a recognition slot while its thread still runs."""
    worker=asyncio.create_task(asyncio.to_thread(function,*args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        # Retrieve exceptions from discarded work without ever applying its result.
        if worker.done() and not worker.cancelled():
            worker.exception()
        raise


def _recognition_rank(result):
    candidates=result.get('candidates') or []
    top=candidates[0] if candidates else {}
    evidence=top.get('evidence') or {}
    return (result.get('status')=='needs_confirmation',
            evidence.get('number_match') is True,
            result.get('ocr',{}).get('number') is not None,
            evidence.get('name_match') is True,
            float(top.get('fusion_score',top.get('similarity',0.0))))


def detect_full_card(image, model_id=None, confidence=.45, size=480, want_corners=False):
    """Detect tightly-cropped full-bleed single card images where edge-to-edge pose fails."""
    w, h = image.width, image.height
    if w < 180 or h < 240 or not (0.62 <= (w / h) <= 0.82):
        return None
    try:
        border = int(max(w, h) * 0.35)
        padded = ImageOps.expand(image, border=border, fill=(32, 32, 32))
        boxes, info = detect(padded, model_id, confidence, size)
        raw = boxes.data if hasattr(boxes, 'data') else np.asarray(boxes, dtype=np.float32)
        rows = np.asarray(raw, dtype=np.float32).reshape(-1, 6) if len(raw) else np.zeros((0, 6), dtype=np.float32)
    except Exception:
        return None
    valid = [r for r in rows if float(r[4]) >= max(0.55, confidence)]
    if len(valid) != 1:
        return None
    x1, y1, x2, y2, conf, _ = [float(v) for v in valid[0]]
    ox1 = max(0.0, x1 - border)
    oy1 = max(0.0, y1 - border)
    ox2 = min(float(w), x2 - border)
    oy2 = min(float(h), y2 - border)
    if (ox2 - ox1) / w < 0.82 or (oy2 - oy1) / h < 0.82:
        return None
    out_rows = np.array([[ox1, oy1, ox2, oy2, conf, 0.0]], dtype=np.float32)
    if want_corners:
        rx2 = min(float(w) - 1e-3, max(ox1 + 1.0, ox2))
        ry2 = min(float(h) - 1e-3, max(oy1 + 1.0, oy2))
        out_corners = np.array([[[ox1, oy1], [rx2, oy1], [rx2, ry2], [ox1, ry2]]], dtype=np.float32)
    else:
        out_corners = None
    return out_rows, out_corners, info


def recognize(image):
    from .pipeline import identify
    first=identify(image,encode_images=False)
    top=first['candidates'][0] if first.get('candidates') else {}
    if first['ocr'].get('number') is not None or (first.get('status')=='needs_confirmation' and top.get('similarity',0)>=.75):
        return first
    second=identify(image.rotate(180),encode_images=False)
    return second if _recognition_rank(second)>_recognition_rank(first) else first


@router.websocket('/v1/live')
async def live(ws:WebSocket):
    global active_sessions
    key=os.getenv('AI_INTERNAL_KEY','')
    if not key or not secrets.compare_digest(ws.headers.get('x-cardlens-key',''),key):
        await ws.close(code=1008);return
    if active_sessions>=2:
        await ws.close(code=1013);return
    active_sessions+=1
    try:
        await ws.accept()
    except BaseException:
        active_sessions-=1
        raise
    queue=asyncio.Queue(maxsize=1);send_lock=asyncio.Lock();tasks=set();tracks={};workers=[]
    policy=DEFAULT_POLICY
    sequence=0;configuration_revision=0;configuration_id=None;epoch=0;paused=False
    settings={'confidence':.45,'size':480,'model':None,'pose_model':None}

    async def send(payload, work_epoch=None):
        async with send_lock:
            if work_epoch is not None and work_epoch!=epoch:return
            if payload['type']=='frame':
                # Recognition can finish while another card is being cropped in a worker.
                for item in payload['tracks']:
                    state=tracks.get(int(item['track_id']))
                    if state is not None and state.generation==item['generation'] and item['quality']['usable']:
                        item.update(status=state.status,candidates=state.candidates,votes=state.votes,ocr=state.ocr)
            elif payload['type']=='identity':
                state=tracks.get(int(payload['track_id']))
                if state is None or state.generation!=payload['generation'] or not state.usable:return
            await ws.send_json(payload)

    def current(track_id, state, generation, work_epoch):
        return (work_epoch==epoch and tracks.get(track_id) is state
                and state.generation==generation and state.usable)

    async def recognition(track_id, state, generation, work_epoch, revision, request_id):
        try:
            async with recognition_gate:
                if not current(track_id,state,generation,work_epoch):return
                best=state.take_best_frame(time.monotonic(),policy)
                if best is None or not state.should_recognize(time.monotonic(),best.sharpness,policy):return
                result=await run_thread(recognize,best.image)
            if not current(track_id,state,generation,work_epoch):return
            state.accept(result,policy,signature=best.signature)
            state.best_quality=max(state.best_quality,best.sharpness)
            state.last_recognized=time.monotonic()
            await send({'type':'identity','configuration_revision':revision,'configuration_id':request_id,
                        'track_id':str(track_id),'generation':generation,
                        'status':state.status,'candidates':state.candidates,'ocr':state.ocr,'votes':state.votes},work_epoch)
        except Exception:
            log.exception('live_recognition_failed track=%s',track_id)
            if current(track_id,state,generation,work_epoch):state.status='unknown'
        finally:
            if state.generation==generation:state.pending=False

    async def receive():
        nonlocal paused,tracker,sequence,settings,configuration_revision,configuration_id,epoch
        while True:
            message=await ws.receive()
            if message['type']=='websocket.disconnect':raise WebSocketDisconnect()
            if message.get('bytes') is not None:
                data=message['bytes']
                if len(data)>MAX_FRAME:await ws.close(code=1009);return
                if paused:continue
                sequence+=1;put_latest(queue,(epoch,sequence,data,time.monotonic()))
            elif message.get('text'):
                try:
                    config=json.loads(message['text'])
                    if not isinstance(config,dict):raise ValueError('invalid_config')
                except (ValueError,TypeError):
                    await send({'type':'error','code':'invalid_config','detail':'invalid_config',
                                'configuration_id':None,'configuration_revision':configuration_revision})
                    continue
                if config.get('type')=='pause':paused=True
                elif config.get('type')=='resume':paused=False
                elif config.get('type')=='configure':
                    attempted_id=config.get('configuration_id')
                    if type(attempted_id) is not int or not 0<=attempted_id<=2**53-1:attempted_id=None
                    try:
                        new_settings,request_id=validated_settings(config)
                        # Fence pending work during boundary I/O, but preserve valid state on rejection.
                        epoch+=1
                        while not queue.empty():queue.get_nowait()
                        for task in list(tasks):task.cancel()
                        pose=await run_thread(verified_pose_metadata)
                        new_settings['pose_model']=configured_pose_model(config,pose['pose_active'])
                        new_session_tracker=new_tracker(new_settings['confidence'])
                    except ValueError as exc:
                        code='pose_detector_not_active' if str(exc)=='pose_detector_not_active' else 'invalid_config'
                        await send({'type':'error','code':code,'detail':code,
                                    'configuration_id':attempted_id,'configuration_revision':configuration_revision})
                        continue
                    configuration_revision+=1;configuration_id=request_id
                    settings=new_settings;tracker=new_session_tracker
                    for state in tracks.values():state.reset()
                    tracks.clear()
                    while not queue.empty():queue.get_nowait()
                    await send({'type':'configured','configuration_revision':configuration_revision,
                                'configuration_id':configuration_id,'pose_model':settings['pose_model'],'tracks':[]},epoch)

    async def process():
        while True:
            work_epoch,seq,raw,received=await queue.get()
            revision=configuration_revision;request_id=configuration_id
            frame_settings=settings.copy();frame_tracker=tracker
            if work_epoch!=epoch:continue
            try:
                dequeued=time.monotonic()
                started=time.perf_counter()
                image=await run_thread(decode_frame,raw)
                if work_epoch!=epoch:continue
                full=await run_thread(detect_full_card,image,frame_settings['model'],frame_settings['confidence'],frame_settings['size'],bool(frame_settings['pose_model']))
                if work_epoch!=epoch:continue
                if full is not None:
                    boxes,corners,info=full
                    if frame_settings['pose_model']:
                        info={**info,'id':frame_settings['pose_model']}
                elif frame_settings['pose_model']:
                    boxes,corners,info=await run_thread(detect_pose,image,frame_settings['pose_model'],frame_settings['confidence'],frame_settings['size'])
                else:
                    boxes,info=await run_thread(detect,image,frame_settings['model'],frame_settings['confidence'],frame_settings['size'])
                    corners=None
                if work_epoch!=epoch:continue
                tracked=await run_thread(track_frame,frame_tracker,boxes,image)
                if work_epoch!=epoch:continue
                output=[];now=time.monotonic()
                visible={int(row[4]) for row in tracked[:24]}
                for ident,state in tracks.items():
                    if ident not in visible:
                        state.usable=False;state.best_frame=None
                for row in tracked[:24]:
                    x1,y1,x2,y2,track_id,score,_,detection_index=row.tolist();track_id=int(track_id)
                    state=tracks.setdefault(track_id,TrackState());state.visible_at=now
                    correction='yolo26_pose' if corners is not None else 'bbox'
                    ordered_corners=None
                    try:
                        if corners is not None:
                            # ByteTrack retains the original index through filtering and Kalman smoothing.
                            index=int(detection_index)
                            if index!=detection_index or not 0<=index<len(corners):raise ValueError('invalid_pose_detection_index')
                            smoothed=state.smooth_corners(corners[index],image.width,image.height,now,policy)
                        else:smoothed=None
                        flat,ordered_corners,q,signature,correction=await run_thread(prepare_crop,image,[x1,y1,x2,y2],smoothed,policy)
                        if work_epoch!=epoch:break
                        if q['usable']:state.observe(signature,now,policy)
                        state.usable=q['usable']
                        if q['usable']:
                            state.offer_frame(flat,signature,q['sharpness'],received,policy)
                        else:state.best_frame=None
                        status=state.status if q['usable'] else 'quality'
                    except (ValueError,IndexError) as exc:
                        state.usable=False;state.best_frame=None
                        q={'usable':False,'reasons':[str(exc)]}
                        status='reject';correction='pose_rejected' if corners is not None else 'crop_rejected'
                    output.append({'track_id':str(track_id),'generation':state.generation,
                        'box':[x1/image.width,y1/image.height,x2/image.width,y2/image.height],
                        'quadrilateral':normalize_corners(ordered_corners,image.width,image.height) if ordered_corners is not None else None,
                        'confidence':score,'status':status,'quality':q,'correction':correction,
                        'candidates':state.candidates if q['usable'] else [],'votes':state.votes if q['usable'] else 0})
                if work_epoch!=epoch:continue
                for ident,state in list(tracks.items()):
                    if state.best_frame is not None and now-state.best_frame.captured_at>policy.best_frame_max_age_seconds:
                        state.best_frame=None
                    if now-state.visible_at>8 and not state.pending:del tracks[ident]
                elapsed=(time.perf_counter()-started)*1000
                await send({'type':'frame','configuration_revision':revision,'configuration_id':request_id,
                            'sequence':seq,'width':image.width,'height':image.height,'tracks':output,
                            'timing_ms':{'detect_track':round(elapsed,2),'queue':round(max(0,dequeued-received)*1000,2)},
                            'model':info['id'],'pose_model':info['id'] if frame_settings['pose_model'] else None,
                            'experimental':not info.get('real_photo_validated',False),
                            'suggested_interval_ms':max(100,min(1000,int(elapsed*1.2)))},work_epoch)
                # Publish track generations before their identity events can reach the browser.
                if work_epoch!=epoch:continue
                prioritized=sorted(output,key=lambda item:(tracks[int(item['track_id'])].votes,tracks[int(item['track_id'])].last_recognized))
                for item in prioritized:
                    track_id=int(item['track_id']);state=tracks[track_id]
                    if (state.usable and state.best_frame is not None and not state.pending and len(tasks)<12
                            and state.should_recognize(time.monotonic(),state.best_frame.sharpness,policy)):
                        state.pending=True
                        task=asyncio.create_task(recognition(track_id,state,state.generation,work_epoch,revision,request_id))
                        tasks.add(task);task.add_done_callback(tasks.discard)
            except (ValueError,FileNotFoundError) as exc:
                await send({'type':'error','detail':str(exc),'configuration_revision':revision,'configuration_id':request_id},work_epoch)
            except Exception:
                log.exception('live_detection_failed')
                await send({'type':'error','detail':'detection_failed','configuration_revision':revision,'configuration_id':request_id},work_epoch)
    try:
        pose=await run_thread(verified_pose_metadata)
        tracker=new_tracker()
        bbox_registry=registry()
        await send({'type':'ready','configuration_revision':configuration_revision,'configuration_id':configuration_id,
                    'registry':{'active':bbox_registry['active'],'pose_active':pose['pose_active']}})
        workers=[asyncio.create_task(receive()),asyncio.create_task(process())]
        done,_=await asyncio.wait(workers,return_when=asyncio.FIRST_COMPLETED)
        for task in done:task.result()
    except (WebSocketDisconnect,ValueError,json.JSONDecodeError):pass
    except Exception:
        log.exception('live_session_failed')
    finally:
        epoch+=1
        for state in tracks.values():state.reset()
        for task in workers+list(tasks):task.cancel()
        await asyncio.gather(*workers,*tasks,return_exceptions=True)
        tracks.clear();active_sessions-=1
        if not active_sessions and os.getenv('PKMLENS_UNLOAD_ON_CLOSE')=='1':await run_thread(unload)
        log.info('live_session_closed remaining=%d',active_sessions)
