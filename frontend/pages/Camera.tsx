import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, CameraSlash, Check, Crosshair, DownloadSimple, Info, Pause, Play, Scan, SlidersHorizontal, X } from '@phosphor-icons/react';
import { api, money, useStore } from '../api';
import type { Candidate, Price, SessionCard } from '../types';
import RecognitionPanel, { type ScanTrack } from '../components/RecognitionPanel';

type Source = 'camera' | 'image' | 'video';
type Detector = { id: string; model?: string; real_photo_validated?: boolean };
type LiveStatus = { models: Detector[]; active: string | null; pose_active?: string | null; pose_models?: Detector[] };
type ConfigurationStamp = { configuration_id: number; configuration_revision: number };
type Frame = ConfigurationStamp & { type: 'frame'; tracks: ScanTrack[]; timing_ms: { detect_track: number }; suggested_interval_ms: number; experimental?: boolean };
type Identity = ConfigurationStamp & { type: 'identity'; track_id: string; generation: string; status: string; candidates: Candidate[]; ocr: ScanTrack['ocr']; votes: number; consensus_required?: number };
type Settings = { model: string; poseModel: string; usePose: boolean; confidence: number };
type Summary = { count: number; p50: number; p95: number; detectorP50: number; detectorP95: number };
const IMAGE_LIMIT = 20 * 1024 * 1024;
const VIDEO_LIMIT = 200 * 1024 * 1024;

function waitForMedia(element: HTMLMediaElement | HTMLImageElement, event: string, signal: AbortSignal, load: () => void) {
  return new Promise<void>((resolve, reject) => {
    const finish = (error?: Error) => {
      clearTimeout(timeout);
      element.removeEventListener(event, loaded); element.removeEventListener('error', failed);
      signal.removeEventListener('abort', aborted);
      if (error) reject(error); else resolve();
    };
    const loaded = () => finish();
    const failed = () => finish(new Error('Không giải mã được tệp. Chọn ảnh PNG/JPEG/WEBP hoặc video MP4/WebM hỗ trợ trong trình duyệt.'));
    const aborted = () => finish(new DOMException('Cancelled', 'AbortError'));
    const timeout = setTimeout(failed, 15000);
    element.addEventListener(event, loaded, { once: true }); element.addEventListener('error', failed, { once: true });
    signal.addEventListener('abort', aborted, { once: true });
    if (signal.aborted) aborted(); else load();
  });
}

export default function CameraPage() {
  const { ready, saved, toggleSaved } = useStore();
  const video = useRef<HTMLVideoElement>(null);
  const image = useRef<HTMLImageElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const objectUrl = useRef<string | null>(null);
  const mediaAbort = useRef<AbortController | null>(null);
  const cancelPreviewMeter = useRef<() => void>(() => {});
  const socket = useRef<WebSocket | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const sendInterval = useRef(250);
  const inFlight = useRef<{ configurationId: number; revision: number; started: number | null } | null>(null);
  const pausedRef = useRef(false);
  const generation = useRef(0);
  const configurationId = useRef(0);
  const configurationRevision = useRef<number | null>(null);
  const configurationStarted = useRef(0);
  const activeSource = useRef<Source | null>(null);
  const settingsRef = useRef<Settings>({ model: '', poseModel: '', usePose: false, confidence: .30 });
  const lastResult = useRef(0);
  const samples = useRef<{ roundtrip: number; detector: number }[]>([]);
  const autoRecorded = useRef<Set<string>>(new Set());
  const [source, setSource] = useState<Source>('camera');
  const [file, setFile] = useState<File | null>(null);
  const [hasPreview, setHasPreview] = useState(false);
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [paused, setPaused] = useState(false);
  const [configuring, setConfiguring] = useState(false);
  const [configurationFailed, setConfigurationFailed] = useState(false);
  const [error, setError] = useState('');
  const [tracks, setTracks] = useState<ScanTrack[]>([]);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [device, setDevice] = useState('');
  const [detectors, setDetectors] = useState<Detector[]>([]);
  const [model, setModel] = useState('');
  const [poseModels, setPoseModels] = useState<Detector[]>([]);
  const [poseModel, setPoseModel] = useState('');
  const [usePose, setUsePose] = useState(false);
  const [confidence, setConfidence] = useState(.30);
  const [autoAddCamera, setAutoAddCamera] = useState(true);
  const [autoAddFile, setAutoAddFile] = useState(false);
  const [selectedId, setSelectedId] = useState('');
  const [latency, setLatency] = useState<number | null>(null);
  const [roundtrip, setRoundtrip] = useState<number | null>(null);
  const [aiFps, setAiFps] = useState<number | null>(null);
  const [previewFps, setPreviewFps] = useState<number | null>(null);
  const [previewMethod, setPreviewMethod] = useState('unavailable');
  const [videoRatio, setVideoRatio] = useState(4 / 3);
  const [session, setSession] = useState<SessionCard[]>([]);
  const [undoSession, setUndoSession] = useState<SessionCard[] | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [events, setEvents] = useState<{ time: string; text: string }[]>([]);
  const [debug, setDebug] = useState(false);
  const [settings, setSettings] = useState(false);
  const selected = tracks.find(track => track.track_id === selectedId)
    || tracks.find(track => track.status === 'needs_confirmation')
    || tracks.find(track => track.status === 'stabilizing')
    || tracks[0];
  const confirmedCount = tracks.filter(track => track.status === 'needs_confirmation' && track.candidates[0]?.card).length;
  const activeDetector = usePose ? poseModels.find(value => value.id === poseModel) : detectors.find(value => value.id === model);
  const poseAvailable = !!poseModel && poseModels.some(value => value.id === poseModel);
  const modelReady = usePose ? poseAvailable : !!model;
  const experiment = !!activeDetector && !activeDetector.real_photo_validated;
  const measurement = (value: number | null) => value == null ? '—' : value.toFixed(1);
  const logEvent = useCallback((text: string) => {
    // Only caller-owned labels/codes enter this bounded log; never media, paths or server payloads.
    setEvents(previous => [...previous.slice(-39), { time: new Date().toLocaleTimeString('vi-VN'), text }]);
  }, []);

  const stop = useCallback((clearPicker = true) => {
    generation.current += 1; activeSource.current = null;
    clearTimeout(timer.current);
    mediaAbort.current?.abort(); mediaAbort.current = null;
    cancelPreviewMeter.current(); cancelPreviewMeter.current = () => {};
    const connection = socket.current;
    socket.current = null; connection?.close();
    stream.current?.getTracks().forEach(track => track.stop()); stream.current = null;
    if (video.current) { video.current.pause(); video.current.srcObject = null; video.current.removeAttribute('src'); video.current.load(); }
    image.current?.removeAttribute('src');
    if (canvas.current) { canvas.current.width = 0; canvas.current.height = 0; }
    if (objectUrl.current) { URL.revokeObjectURL(objectUrl.current); objectUrl.current = null; }
    if (clearPicker && fileInput.current) fileInput.current.value = '';
    configurationRevision.current = null; inFlight.current = null; pausedRef.current = false; lastResult.current = 0;
    autoRecorded.current.clear();
    setFile(null); setHasPreview(false); setRunning(false); setStarting(false); setPaused(false); setTracks([]); setSelectedId('');
    setConfiguring(false); setConfigurationFailed(false); setAiFps(null); setPreviewFps(null); setPreviewMethod('unavailable');
  }, []);

  useEffect(() => {
    if (!ready) return;
    const controller = new AbortController();
    api<LiveStatus>('/live/status', { signal: controller.signal }).then(result => {
      if (controller.signal.aborted) return;
      setDetectors(result.models); setModel(result.active || ''); setPoseModel(result.pose_active || ''); setPoseModels(result.pose_models || []);
      settingsRef.current = { ...settingsRef.current, model: result.active || '', poseModel: result.pose_active || '' };
      if (!result.active) setError('Chưa có model sẵn sàng. Kiểm tra dịch vụ AI hoặc chọn model trong công cụ train.');
    }).catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    return () => { controller.abort(); stop(); };
  }, [ready, stop]);

  function measurePreview(v: HTMLVideoElement, currentGeneration: number) {
    let callback = 0, fallback: ReturnType<typeof setInterval> | undefined;
    let lastTime = 0, lastFrames = 0;
    const measure = (now: number, frames: number) => {
      if (!Number.isFinite(frames) || frames <= 0) { lastTime = 0; setPreviewFps(null); return; }
      if (v.paused || v.ended) { lastTime = 0; setPreviewFps(null); return; }
      if (!lastTime || frames < lastFrames) { lastTime = now; lastFrames = frames; return; }
      if (now - lastTime >= 900) {
        setPreviewFps(Math.round((frames - lastFrames) * 10000 / (now - lastTime)) / 10);
        lastTime = now; lastFrames = frames;
      }
    };
    if (typeof v.requestVideoFrameCallback === 'function') {
      setPreviewMethod('requestVideoFrameCallback / presentedFrames');
      const next = (now: number, metadata: VideoFrameCallbackMetadata) => {
        if (generation.current !== currentGeneration) return;
        measure(now, metadata.presentedFrames); callback = v.requestVideoFrameCallback(next);
      };
      callback = v.requestVideoFrameCallback(next);
    } else if (typeof v.getVideoPlaybackQuality === 'function') {
      setPreviewMethod('playbackQuality / total minus dropped frames');
      fallback = setInterval(() => {
        if (generation.current !== currentGeneration) return;
        const quality = v.getVideoPlaybackQuality();
        measure(performance.now(), quality.totalVideoFrames - quality.droppedVideoFrames);
      }, 1000);
    } else { setPreviewMethod('unavailable'); setPreviewFps(null); }
    cancelPreviewMeter.current = () => { if (callback) v.cancelVideoFrameCallback(callback); clearInterval(fallback); };
  }

  function sendConfiguration(ws: WebSocket, next: Settings) {
    configurationId.current += 1; configurationRevision.current = null; configurationStarted.current = performance.now();
    inFlight.current = null; lastResult.current = 0;
    setTracks([]); setSelectedId(''); setConfiguring(true); setConfigurationFailed(false); setError('');
    setAiFps(null); setLatency(null); setRoundtrip(null); samples.current = []; setSummary(null);
    ws.send(JSON.stringify({ type: 'configure', configuration_id: configurationId.current, model: next.model || null,
      pose_model: next.usePose ? next.poseModel : null, use_pose: next.usePose, confidence: next.confidence, size: next.usePose ? 480 : 640 }));
    logEvent('Configuration requested #' + configurationId.current + ' / ' + (next.usePose ? 'pose' : 'bbox'));
  }

  async function start() {
    const picked = file;
    stop(false); setError(''); setStarting(true); setSummary(null); samples.current = []; sendInterval.current = 250;
    const currentGeneration = generation.current; activeSource.current = source;
    const controller = new AbortController(); mediaAbort.current = controller;
    logEvent('Source starting: ' + source);
    try {
      const v = video.current, img = image.current;
      if (!v || !img) throw new Error('Preview unavailable');
      if (source === 'camera') {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera cần localhost hoặc HTTPS.');
        const media = await navigator.mediaDevices.getUserMedia({ video: {
          deviceId: device ? { exact: device } : undefined, width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: 'environment' },
        }, audio: false });
        if (currentGeneration !== generation.current) { media.getTracks().forEach(track => track.stop()); return; }
        stream.current = media; v.srcObject = media; await v.play();
        if (currentGeneration !== generation.current) return;
        // Device enumeration must not hold up or revive a stopped camera.
        void navigator.mediaDevices.enumerateDevices().then(values => {
          if (currentGeneration === generation.current) setDevices(values.filter(value => value.kind === 'videoinput'));
        }).catch(() => { if (currentGeneration === generation.current) logEvent('Camera list unavailable'); });
      } else {
        if (!picked) throw new Error('Chọn tệp trước khi quét.');
        setFile(picked); objectUrl.current = URL.createObjectURL(picked);
        const url = objectUrl.current;
        if (source === 'image') {
          await waitForMedia(img, 'load', controller.signal, () => { img.src = url; });
          if (currentGeneration !== generation.current) return;
          if (img.naturalWidth * img.naturalHeight > 40000000) throw new Error('Ảnh vượt giới hạn 40 megapixel. Hãy giảm kích thước.');
        } else {
          await waitForMedia(v, 'loadeddata', controller.signal, () => { v.src = url; v.load(); });
          if (currentGeneration !== generation.current) return;
          await v.play();
        }
      }
      if (currentGeneration !== generation.current) return;
      const width = source === 'image' ? img.naturalWidth : v.videoWidth;
      const height = source === 'image' ? img.naturalHeight : v.videoHeight;
      if (!width || !height) throw new Error('Nguồn không có khung hình hợp lệ.');
      setVideoRatio(width / height); setHasPreview(true);
      if (source !== 'image') measurePreview(v, currentGeneration);
      const defaultWs = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/v1/ws/scan`;
      const envWs = import.meta.env.VITE_WS_URL || (import.meta.env.VITE_API_URL ? import.meta.env.VITE_API_URL.replace(/^http/, 'ws').replace(/\/+$/, '') + '/api/v1/ws/scan' : '');
      const token = localStorage.getItem('pkmlens-session-token') || '';
      const targetWs = envWs ? envWs + (token ? `?token=${encodeURIComponent(token)}` : '') : defaultWs;
      const ws = new WebSocket(targetWs);
      socket.current = ws;
      let connectionError = false;
      ws.onopen = () => {
        if (socket.current !== ws || currentGeneration !== generation.current) return;
        setRunning(true); setStarting(false); sendConfiguration(ws, settingsRef.current);
        const sendFrame = () => {
          if (socket.current !== ws || currentGeneration !== generation.current) return;
          const c = canvas.current;
          const revision = configurationRevision.current, id = configurationId.current;
          if (configurationStarted.current && performance.now() - configurationStarted.current > 15000 && revision == null) {
            configurationStarted.current = 0; setConfiguring(false); setConfigurationFailed(true);
            setError('Dịch vụ chưa xác nhận cấu hình. Chọn lại cài đặt hoặc khởi động quét lại.'); logEvent('Configuration acknowledgement timeout');
          }
          if (inFlight.current?.started && performance.now() - inFlight.current.started > 15000) {
            stop(); setError('Không nhận được khung AI sau 15 giây. Hãy thử quét lại.'); logEvent('AI frame timeout'); return;
          }
          const mediaReady = source === 'image' ? img.complete && !!img.naturalWidth : v.readyState >= 2 && !v.ended;
          if (!pausedRef.current && revision != null && !inFlight.current && c && mediaReady
              && ws.readyState === WebSocket.OPEN && ws.bufferedAmount < 500000) {
            const frameWidth = source === 'image' ? img.naturalWidth : v.videoWidth;
            const frameHeight = source === 'image' ? img.naturalHeight : v.videoHeight;
            const scale = Math.min(1, 1280 / Math.max(frameWidth, frameHeight));
            c.width = Math.max(1, Math.round(frameWidth * scale)); c.height = Math.max(1, Math.round(frameHeight * scale));
            const pending = { configurationId: id, revision, started: null as number | null };
            inFlight.current = pending;
            try {
              const context = c.getContext('2d');
              if (!context) throw new Error('Canvas unavailable');
              context.drawImage(source === 'image' ? img : v, 0, 0, c.width, c.height);
              c.toBlob(blob => {
                if (inFlight.current !== pending) return;
                if (blob && socket.current === ws && generation.current === currentGeneration && !pausedRef.current
                    && configurationId.current === id && configurationRevision.current === revision && ws.readyState === WebSocket.OPEN) {
                  pending.started = performance.now(); ws.send(blob);
                } else { inFlight.current = null; if (!blob) logEvent('Frame encode failed'); }
              }, 'image/jpeg', .82);
            } catch {
              stop(); setError('Không mã hóa được khung hình. Hãy chọn nguồn khác.'); logEvent('Frame encode failed'); return;
            }
          }
          timer.current = setTimeout(sendFrame, sendInterval.current);
        };
        sendFrame();
      };
      ws.onmessage = event => {
        if (socket.current !== ws || currentGeneration !== generation.current) return;
        let value: Frame | Identity | (ConfigurationStamp & { type: 'configured'; pose_model: string | null })
          | { type: 'error'; detail: string; configuration_id?: number; configuration_revision?: number } | { type: 'ready' };
        try { value = JSON.parse(event.data); } catch { logEvent('Invalid AI message'); return; }
        if (value.type === 'ready') return;
        if (value.configuration_id !== configurationId.current) return;
        if (value.type === 'configured') {
          if (!Number.isInteger(value.configuration_revision) || value.configuration_revision < 1
              || (configurationRevision.current != null && value.configuration_revision !== configurationRevision.current)) return;
          const desired = settingsRef.current.usePose ? settingsRef.current.poseModel : null;
          if (value.pose_model !== desired) { configurationStarted.current = 0; setConfiguring(false); setConfigurationFailed(true); setError('Model AI xác nhận không khớp lựa chọn. Hãy chọn lại.'); return; }
          configurationRevision.current = value.configuration_revision; configurationStarted.current = 0;
          setConfiguring(false); setConfigurationFailed(false); connectionError = false; logEvent('Configuration accepted #' + value.configuration_id); return;
        }
        if (value.type === 'error') {
          if (configurationRevision.current != null && value.configuration_revision !== configurationRevision.current) return;
          inFlight.current = null; connectionError = true; setTracks([]); setSelectedId('');
          configurationRevision.current = null; configurationStarted.current = 0; setConfiguring(false); setConfigurationFailed(true);
          const labels: Record<string, string> = {
            pose_detector_not_active: 'Pose chưa sẵn sàng. Tắt pose để dùng model bbox.',
            invalid_config: 'Cấu hình bị từ chối. Hãy chọn lại cài đặt.',
            card_detector_not_activated: 'Chưa có model YOLO được kích hoạt.',
          };
          const code = Object.hasOwn(labels, value.detail) ? value.detail : 'inference_failed';
          setError(labels[code] || 'Máy quét gặp lỗi. Chọn lại cài đặt hoặc thử quét lại.'); logEvent('AI error: ' + code); return;
        }
        if (value.configuration_revision !== configurationRevision.current) return;
        if (pausedRef.current) { if (value.type === 'frame') inFlight.current = null; return; }
        if (value.type === 'frame') {
          const pending = inFlight.current; inFlight.current = null;
          setTracks(previous => value.tracks.map(track => {
            const old = previous.find(item => item.track_id === track.track_id && item.generation === track.generation);
            return old?.ocr ? { ...track, ocr: old.ocr } : track;
          }));
          const now = performance.now();
          if (lastResult.current) setAiFps(Math.round(10000 / (now - lastResult.current)) / 10);
          lastResult.current = now;
          const detector = value.timing_ms.detect_track;
          setLatency(Number.isFinite(detector) ? detector : null);
          if (pending?.started && pending.configurationId === value.configuration_id && pending.revision === value.configuration_revision && Number.isFinite(detector)) {
            const elapsed = now - pending.started; setRoundtrip(elapsed);
            samples.current = [...samples.current.slice(-599), { roundtrip: elapsed, detector }];
            const times = samples.current.map(item => item.roundtrip).sort((a, b) => a - b);
            const processing = samples.current.map(item => item.detector).sort((a, b) => a - b);
            const percentile = (values: number[], percent: number) => Math.round(values[Math.max(0, Math.ceil(values.length * percent) - 1)]);
            setSummary({ count: times.length, p50: percentile(times, .5), p95: percentile(times, .95), detectorP50: percentile(processing, .5), detectorP95: percentile(processing, .95) });
          }
          sendInterval.current = Math.max(100, Math.min(2000, value.suggested_interval_ms || 200));
        } else if (value.type === 'identity') {
          setTracks(previous => previous.map(track => track.track_id === value.track_id && track.generation === value.generation
            ? { ...track, candidates: value.candidates, status: value.status, ocr: value.ocr, votes: value.votes, consensus_required: value.consensus_required ?? track.consensus_required } : track));
        }
      };
      ws.onerror = () => {
        if (socket.current !== ws) return;
        connectionError = true; stop();
        const isStaticHost = !import.meta.env.VITE_WS_URL && !import.meta.env.VITE_API_URL && location.hostname.includes('vercel.app');
        setError(isStaticHost ? 'Chưa kết nối máy chủ AI Python. Cần cấu hình VITE_API_URL hoặc VITE_WS_URL trên Vercel để quét thẻ.' : 'Chưa kết nối được dịch vụ AI. Kiểm tra máy chủ và thử lại.');
        logEvent('AI socket error');
      };
      ws.onclose = () => {
        if (socket.current !== ws) return;
        stop(); if (!connectionError) setError('Kết nối máy quét đã đóng. Bạn có thể quét lại.'); logEvent('AI socket closed');
      };
    } catch (reason) {
      if (currentGeneration !== generation.current) return;
      const name = (reason as Error).name;
      const message = name === 'NotAllowedError'
        ? 'Camera chưa được cho phép. Cấp quyền camera trong trình duyệt rồi bấm bật lại.'
        : name === 'NotFoundError' ? 'Không tìm thấy camera. Kết nối webcam và thử lại.' : name === 'NotSupportedError' ? 'Trình duyệt không hỗ trợ codec video này. Chọn MP4/WebM khác.' : (reason as Error).message;
      stop(); setError(message); logEvent('Source start failed: ' + source);
    }
  }

  function chooseFile(picked?: File) {
    stop(false); setError('');
    if (!picked) return;
    const extension = picked.name.split('.').pop()?.toLowerCase() || '';
    const types: Record<string, string[]> = source === 'image'
      ? { png: ['image/png'], jpg: ['image/jpeg'], jpeg: ['image/jpeg'], webp: ['image/webp'] }
      : { mp4: ['video/mp4'], webm: ['video/webm'] };
    if (!types[extension] || (picked.type && !types[extension].includes(picked.type))) {
      if (fileInput.current) fileInput.current.value = '';
      setError('Chỉ nhận PNG/JPEG/WEBP cho ảnh, MP4/WebM cho video.'); logEvent('File rejected: unsupported type'); return;
    }
    if (!picked.size || picked.size > (source === 'image' ? IMAGE_LIMIT : VIDEO_LIMIT)) {
      if (fileInput.current) fileInput.current.value = '';
      setError(source === 'image' ? 'Ảnh phải từ 1 byte đến 20 MiB.' : 'Video phải từ 1 byte đến 200 MiB.'); logEvent('File rejected: size limit'); return;
    }
    setFile(picked); logEvent('Local file selected: ' + source);
  }
  function togglePause() {
    const ws = socket.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    pausedRef.current = !pausedRef.current; setPaused(pausedRef.current); lastResult.current = 0; setAiFps(null);
    ws.send(JSON.stringify({ type: pausedRef.current ? 'pause' : 'resume' }));
    if (pausedRef.current) {
      // Cancel an encode that has not been sent; an outstanding sent frame remains fenced until received.
      if (inFlight.current?.started == null) inFlight.current = null;
      if (source === 'video') { video.current?.pause(); cancelPreviewMeter.current(); setPreviewFps(null); }
    } else if (source === 'video' && video.current) {
      const currentGeneration = generation.current, media = video.current;
      void media.play().then(() => {
        if (generation.current === currentGeneration && !pausedRef.current) measurePreview(media, currentGeneration);
      }).catch(() => {
        if (generation.current === currentGeneration) { stop(); setError('Không tiếp tục được video. Chọn lại tệp.'); }
      });
    }
    logEvent(pausedRef.current ? 'Inference paused' : 'Inference resumed');
  }
  function configure(nextModel: string, nextConfidence: number, nextUsePose = usePose) {
    const next = { model: nextModel, confidence: nextConfidence, usePose: nextUsePose && poseAvailable, poseModel };
    settingsRef.current = next; setModel(nextModel); setConfidence(nextConfidence); setUsePose(next.usePose); setTracks([]); setSelectedId('');
    if (socket.current?.readyState === WebSocket.OPEN) sendConfiguration(socket.current, next);
  }
  const autoSaveActive = source === 'camera' ? autoAddCamera : autoAddFile;
  const sessionTotalVnd = session.reduce((sum, item) => sum + (item.priceVnd ?? 0), 0);

  function recordAndFetchPrice(entryKey: string, cardId: string, variant: string | null | undefined, similarity: number | undefined, autoAdded: boolean) {
    if (!cardId) return;
    void api<{ scan_id: string; variant: string; variant_label: string; quote: Price; added_to_collection: boolean }>('/scan/record', {
      method: 'POST',
      body: JSON.stringify({ card_id: cardId, variant: variant || null, condition: 'NM', similarity: similarity ?? null, source, auto_added: autoAdded, fetch_price: true }),
    }).then(res => {
      setSession(previous => previous.map(item => (item.trackKey === entryKey && item.card_id === cardId)
        ? { ...item, variant: item.variant ?? res.variant, variantLabel: item.variant ? item.variantLabel : res.variant_label, priceVnd: res.quote?.amount_vnd ?? item.priceVnd, priceSource: res.quote?.source ?? item.priceSource }
        : item));
      if (res.quote?.amount_vnd != null) logEvent('Fetched live price: ' + money(res.quote.amount_vnd));
    }).catch(() => { /* Keep local session entry usable even if offline. */ });
  }

  useEffect(() => {
    if (!running || !autoSaveActive) return;
    for (const track of tracks) {
      const best = track.candidates[0];
      if (track.status !== 'needs_confirmation' || !best?.card) continue;
      const trackKey = track.track_id + ':' + track.generation;
      const dedupeKey = trackKey + ':' + best.card_id;
      if (autoRecorded.current.has(dedupeKey)) continue;
      autoRecorded.current.add(dedupeKey);
      if (!saved.includes(best.card_id)) toggleSaved(best.card_id);
      const initialPrice = best.card.reference_price?.amount_vnd ?? null;
      const initialSource = best.card.reference_price?.source ?? null;
      const newEntry: SessionCard = {
        card_id: best.card_id, card: best.card, similarity: best.similarity, evidence: best.evidence,
        variant: null, variantLabel: 'Tự động lưu', source, stillImageRepeated: source === 'image',
        trackKey, autoAdded: true, priceVnd: initialPrice, priceSource: initialSource,
      };
      setSession(previous => {
        const exists = previous.some(item => item.trackKey === trackKey && item.card_id === best.card_id);
        return exists ? previous : [...previous, newEntry];
      });
      logEvent('Auto-added recognized card: ' + best.card.name);
      recordAndFetchPrice(trackKey, best.card_id, null, best.similarity, true);
    }
  }, [tracks, running, autoSaveActive, source, saved, toggleSaved, logEvent]);

  function saveCard(card: SessionCard) {
    const trackKey = selected ? selected.track_id + ':' + selected.generation : undefined;
    const entry: SessionCard = { ...card, source, stillImageRepeated: source === 'image', trackKey, autoAdded: false };
    setUndoSession(session);
    const existingIndex = trackKey && card.card_id
      ? session.findIndex(item => item.trackKey === trackKey && item.card_id === card.card_id && item.autoAdded)
      : -1;
    if (existingIndex >= 0) {
      const next = session.slice();
      next[existingIndex] = { ...session[existingIndex], ...entry };
      setSession(next);
    } else {
      setSession([...session, entry]);
    }
    if (card.card_id && !saved.includes(card.card_id)) toggleSaved(card.card_id);
    if (card.card_id && autoSaveActive && trackKey) {
      recordAndFetchPrice(trackKey, card.card_id, card.variant, card.similarity, false);
    }
    logEvent('User confirmed one session card');
  }
  function saveAllConfirmed() {
    const readyTracks = tracks.filter(track => track.status === 'needs_confirmation' && track.candidates[0]?.card);
    if (!readyTracks.length) return;
    setUndoSession(session);
    const additions: SessionCard[] = [];
    for (const track of readyTracks) {
      const best = track.candidates[0];
      const trackKey = track.track_id + ':' + track.generation;
      if (session.some(item => item.trackKey === trackKey && item.card_id === best.card_id)) continue;
      if (!saved.includes(best.card_id)) toggleSaved(best.card_id);
      additions.push({
        card_id: best.card_id, card: best.card,
        similarity: best.similarity, evidence: best.evidence,
        variant: null, variantLabel: 'Chưa xác minh', source, stillImageRepeated: source === 'image',
        trackKey, priceVnd: best.card?.reference_price?.amount_vnd ?? null, priceSource: best.card?.reference_price?.source ?? null,
      });
      if (autoSaveActive) recordAndFetchPrice(trackKey, best.card_id, null, best.similarity, false);
    }
    if (additions.length) setSession([...session, ...additions]);
    logEvent('User confirmed ' + readyTracks.length + ' session cards');
  }
  function removeCard(index: number) { setUndoSession(session); setSession(session.filter((_, i) => i !== index)); }
  function undoLast() { if (undoSession) { setSession(undoSession); setUndoSession(null); logEvent('Last session action undone'); } }
  function exportSession() {
    const cell = (value: string) => '"' + (/^(?:[\t\r\n]|\s*[=+\-@])/.test(value) ? "'" : '') + value.replaceAll('"', '""') + '"';
    const csv = 'card_id,name,set,number,status,source,evidence,variant_id,variant_label,variant_source,variant_verified\n' + session.map(value => [
      value.card_id, value.card?.name || value.ocrName || '', value.card?.set_name || '',
      value.card?.number || value.ocrNumber || '', value.card_id ? 'user_confirmed' : 'unknown_ocr', value.source || 'camera', value.stillImageRepeated ? 'still_image_repeated' : 'user_confirmed_no_accuracy_claim',
      value.variant ?? '', value.variantLabel ?? (value.card_id ? 'Chưa xác minh' : ''), value.variantSource ?? '', 'false',
    ].map(cell).join(',')).join('\n');
    const url = URL.createObjectURL(new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = 'pkmlens-scan-session.csv'; link.click(); URL.revokeObjectURL(url);
  }

  return <>
    <div className="page-heading scanner-heading"><div><h1>Khám phá lá thẻ trong tay.</h1>
      <p>Đưa thẻ vào camera. PkmLens sẽ giúp bạn tìm đúng bản in.</p></div>
      <span className={'connection-state' + (running && !paused && !configuring && !configurationFailed ? ' connected' : '')}><i />
        {starting ? 'Đang mở nguồn' : running ? paused ? 'Tạm dừng' : configuring ? 'Chờ cấu hình' : configurationFailed ? 'Cấu hình lỗi' : 'Đang quét' : 'Sẵn sàng'}</span>
    </div>
    {error && <div role="alert" className="error-state">{error}<button className="icon-button" onClick={() => setError('')} aria-label="Ẩn thông báo"><X size={19} /></button></div>}
    <div className="scanner-layout"><section className="camera-section">
      <div className="camera-source-controls">
        <label htmlFor="scan-source">Nguồn quét<select id="scan-source" value={source} onChange={event => {
          stop(); setSource(event.target.value as Source); setError(''); logEvent('Source changed');
        }}><option value="camera">Camera trực tiếp</option><option value="image">Ảnh cục bộ</option><option value="video">Video cục bộ</option></select></label>
        {source !== 'camera' && <label htmlFor="scan-source-file">{source === 'image' ? 'PNG / JPEG / WEBP · tối đa 20 MiB, 40 MP' : 'MP4 / WebM · tối đa 200 MiB'}
          <input ref={fileInput} id="scan-source-file" type="file" accept={source === 'image' ? '.png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp' : '.mp4,.webm,video/mp4,video/webm'} onChange={event => chooseFile(event.target.files?.[0])} />
          {file && <small className="camera-file-name">{file.name}</small>}
        </label>}
      </div>
      <div className="camera-stage">
        {running && <>
          <div className="camera-topline"><span><i />{paused ? 'TẠM DỪNG' : source === 'camera' ? 'LIVE CAMERA' : source === 'image' ? 'STILL IMAGE · REPEATED' : 'LOCAL VIDEO'}</span>
            <span>Preview {source === 'image' ? 'n/a' : measurement(previewFps)} FPS · AI {measurement(aiFps)} updates/s</span></div>
          {debug && <div className="camera-debug-overlay">
            <span>Detector + track {measurement(latency)} ms · AI roundtrip {measurement(roundtrip)} ms</span>
            <span>Config #{configurationId.current} / revision {configurationRevision.current ?? '—'} · {usePose ? 'pose' : 'bbox'}</span>
            <span>Preview meter: {source === 'image' ? 'still image; no video FPS' : previewMethod}</span>
          </div>}
          {!tracks.length && <div className="camera-wait" role="status"><Crosshair size={21} />{configuring ? 'Đang chờ AI xác nhận cấu hình' : configurationFailed ? 'Cấu hình chưa sẵn sàng · xem thông báo lỗi' : source === 'camera' ? 'Đặt thẻ vào khung hình để nhận diện' : 'Đang tìm thẻ trong tệp'}</div>}
        </>}
        <div className="camera-viewport" style={hasPreview ? { aspectRatio: videoRatio, width: `min(100%, calc(min(68vh, 540px) * ${videoRatio}))` } : undefined}>
          <video ref={video} hidden={source === 'image'} muted playsInline autoPlay onResize={() => {
            const media = video.current;
            if (activeSource.current && media?.videoWidth && media.videoHeight) setVideoRatio(media.videoWidth / media.videoHeight);
          }} onEnded={() => {
            if (activeSource.current === 'video' && video.current?.ended) { stop(); logEvent('Video ended: inference stopped and source released'); }
          }} /><img ref={image} hidden={source !== 'image'} alt="Ảnh cục bộ đang quét" /><canvas ref={canvas} hidden />
          {!running && <div className="camera-placeholder">
            <div className="camera-guide"><Scan size={57} weight="light" /></div>
            <h2>Một lá thẻ, cả câu chuyện.</h2><p>{source === 'camera' ? 'Bật camera và đặt lá thẻ trong khung hình.' : 'Chọn tệp rồi bấm quét để gửi khung hình đến AI.'}<br />Giữ thẻ rõ nét, tránh ánh sáng phản chiếu.</p>
            <button className="button camera-start" onClick={start} disabled={starting || !ready || !modelReady || (source !== 'camera' && !file)}>
              <Camera size={21} />{starting ? 'Đang kết nối…' : source === 'camera' ? 'Bật camera để bắt đầu' : source === 'image' ? 'Quét ảnh đã chọn' : 'Quét video đã chọn'}</button>
            <small><CameraSlash size={14} />Hình ảnh quét không được lưu lại.</small>
            {experiment && <p className="camera-experiment-inline"><Info size={15} />{usePose ? 'Model pose thử nghiệm' : 'Model bbox thử nghiệm'} · chưa kiểm chứng camera thật.</p>}
          </div>}
          {running && <>
            <svg className="tracking-quadrilaterals" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">{tracks.filter(track => track.quadrilateral?.length === 4).map(track => <polygon key={track.track_id + track.generation} points={track.quadrilateral!.map(point => `${point[0]},${point[1]}`).join(' ')} className={track.track_id === selected?.track_id ? 'selected' : ''} />)}</svg>
            {tracks.map(track => <button key={track.track_id + track.generation}
              aria-label={'Chọn thẻ ' + (track.status === 'needs_confirmation' ? track.candidates[0]?.card?.name : '#' + track.track_id)}
              className={'tracking-box ' + (track.status === 'needs_confirmation' ? 'recognized' : '') + (track.quadrilateral?.length === 4 ? ' has-quad' : '')}
              style={{ left: `${track.box[0] * 100}%`, top: `${track.box[1] * 100}%`, width: `${(track.box[2] - track.box[0]) * 100}%`, height: `${(track.box[3] - track.box[1]) * 100}%` }}
              onClick={() => setSelectedId(track.track_id)}>
              <span>{(track.status === 'needs_confirmation' && track.candidates[0]?.card
                ? `${track.candidates[0].card.name}${track.candidates[0].card.number ? ' #' + track.candidates[0].card.number : ''} · sim ${track.candidates[0].similarity.toFixed(2)}`
                : track.status === 'stabilizing' && track.candidates[0]?.card
                ? `${track.candidates[0].card.name}${track.candidates[0].card.number ? ' #' + track.candidates[0].card.number : ''} (${track.votes}/${track.consensus_required ?? 3})`
                : 'Thẻ #' + track.track_id)} · detector {track.confidence.toFixed(2)}</span><i /><i /><i /><i />
            </button>)}
          </>}
        </div>
      </div>
      <div className="camera-bottom"><div><span className="camera-lens-dot" />{running ? `${tracks.length} lá thẻ trong khung${confirmedCount ? ` · ${confirmedCount} đã nhận diện` : ''}` : source === 'camera' ? 'Webcam / camera điện thoại' : source === 'image' ? 'Ảnh cục bộ' : 'Video cục bộ'}</div><div>
        {running && <button className="icon-button" onClick={togglePause} aria-label={paused ? 'Tiếp tục quét' : 'Tạm dừng quét'}>{paused ? <Play size={20} /> : <Pause size={20} />}</button>}
        <button className="icon-button" aria-label="Cài đặt camera" aria-expanded={settings} onClick={() => setSettings(!settings)}><SlidersHorizontal size={21} /></button>
        {(running || starting) && <button className="button stop-camera" onClick={() => { stop(); logEvent('Source stopped by user'); }}><CameraSlash size={18} />{source === 'camera' ? 'Dừng camera' : 'Dừng quét'}</button>}
      </div></div>
      {settings && <div className="camera-settings">
        <label>Camera<select value={device} onChange={event => { stop(); setDevice(event.target.value); }}><option value="">Camera mặc định</option>{devices.map(value => <option value={value.deviceId} key={value.deviceId}>{value.label || 'Camera'}</option>)}</select></label>
        <label>Model YOLO<select value={model} onChange={event => configure(event.target.value, confidence)}>{detectors.map(value => <option value={value.id} key={value.id}>{value.model || value.id} {value.real_photo_validated ? '' : '· thử nghiệm'}</option>)}</select></label>
        <label>Ngưỡng phát hiện · {Math.round(confidence * 100)}%<input type="range" min=".2" max=".9" step=".05" value={confidence} onChange={event => configure(model, Number(event.target.value))} /></label>
        {poseAvailable && <label className="camera-pose-toggle"><input type="checkbox" checked={usePose} onChange={event => configure(model, confidence, event.target.checked)} />Pose keypoint quadrilateral (experimental)</label>}
        <label className="camera-pose-toggle"><input type="checkbox" checked={autoSaveActive} onChange={event => source === 'camera' ? setAutoAddCamera(event.target.checked) : setAutoAddFile(event.target.checked)} />Tự động thêm vào hệ thống &amp; lấy giá online</label>
        <label className="camera-pose-toggle"><input type="checkbox" checked={debug} onChange={event => setDebug(event.target.checked)} />Hiện debug và số đo phiên</label>
      </div>}
      {source === 'image' && <p className="camera-source-note">Ảnh tĩnh được gửi lặp để đối chiếu consensus; đây không phải các khung camera thật độc lập. Bạn vẫn cần xác nhận bản in.</p>}
      {source !== 'camera' && <p className="camera-source-note">Tệp chỉ đọc trong trình duyệt; khung hình gửi qua kết nối AI đã xác thực, không lưu ảnh/video. Dừng hoặc đổi nguồn sẽ giải phóng tệp; video kết thúc sẽ dừng quét.</p>}
      {debug && <details className="camera-diagnostics" open>
        <summary>Đo cục bộ · phiên quét hiện tại</summary>
        {summary ? <p>{summary.count} mẫu gần nhất (tối đa 600, cùng cấu hình) · AI roundtrip p50 {summary.p50} ms / p95 {summary.p95} ms · detector + track p50 {summary.detectorP50} ms / p95 {summary.detectorP95} ms.</p> : <p>Chưa có mẫu roundtrip AI.</p>}
        <p>Roundtrip: từ lúc gửi JPEG đến kết quả frame; không gộp OCR/identity bất đồng bộ. Preview FPS và detector + track là số đo riêng. Không phải kiểm chứng độ chính xác camera thật.</p>
        <ol aria-label="Sự kiện máy quét gần đây">{events.map((event, index) => <li key={index}><time>{event.time}</time> {event.text}</li>)}</ol>
      </details>}
      <div className="scanner-hints"><span><Check size={16} />Đặt thẻ trên nền phẳng</span><span><Check size={16} />Đủ sáng, không lóa</span><span><Check size={16} />Có thể quét nhiều thẻ</span></div>
      {experiment && <div className="experimental-note"><Info size={18} /><p>Model {activeDetector?.model || activeDetector?.id} ({usePose ? 'pose' : 'bbox'}) chưa được kiểm chứng bằng ảnh camera thật. Khả năng nhận diện trên camera thật đang được kiểm chứng; bạn luôn có thể đối chiếu và sửa kết quả.</p></div>}
    </section>
      <RecognitionPanel tracks={tracks} selected={selected} onSelect={setSelectedId} onSave={saveCard} stillImage={source === 'image'} />
    </div>
    <section className="scan-session panel"><div><h2>Phiên quét của bạn <span>{session.length}</span>{sessionTotalVnd > 0 && <> · <span>{money(sessionTotalVnd)}</span></>}</h2><p>{autoSaveActive ? 'Thẻ nhận diện xong sẽ tự động lưu vào hệ thống và lấy giá tham khảo trực tuyến.' : 'Chỉ những lá thẻ bạn xác nhận mới được lưu vào phiên.'}</p></div>
      <div className="session-actions">
      {confirmedCount > 1 && <button className="button secondary" onClick={saveAllConfirmed}><Check size={18} />Lưu tất cả ({confirmedCount} thẻ)</button>}
      <button className="button secondary" disabled={!undoSession} onClick={undoLast}>Hoàn tác thao tác cuối</button>
      <button className="button secondary" disabled={!session.length} onClick={exportSession}><DownloadSimple size={19} />Xuất CSV</button></div>
      {session.length > 0 && <div className="session-cards">{session.map((value, index) => <div key={index}>
        <span>{index + 1}</span><strong>{value.card?.name || value.ocrName || 'Thẻ chưa định danh'}</strong>
        <small>{value.card ? `${value.card.set_name} · #${value.card.number} · ${value.variantLabel ?? 'Chưa xác minh'}` : `OCR chưa xác minh · ${value.ocrNumber || 'Chưa có số thẻ'}`}{value.priceVnd != null ? ` · ${money(value.priceVnd)}${value.priceSource ? ' (' + value.priceSource + ')' : ''}` : ''}{value.stillImageRepeated ? ' · Ảnh tĩnh lặp' : ''}</small>
        <button className="icon-button" aria-label={'Xóa ' + (value.card?.name || value.ocrName || 'thẻ') + ' khỏi phiên'} onClick={() => removeCard(index)}><X size={16} /></button>
      </div>)}</div>}
    </section>
  </>;
}
