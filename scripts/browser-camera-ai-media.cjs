const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(process.env.PLAYWRIGHT_MODULE || 'D:/Project/CardLens/node_modules/playwright'); }

const root = process.cwd();
const imagePath = path.resolve('data/evaluation/camera-scene.jpg');
const videoPath = path.resolve('data/evaluation/ai-media-repeated-scene.mp4');
const reportPath = path.resolve('reports/browser-ai-media-verification.json');
const screenshotDir = path.resolve('.impeccable/review');
const screenshotPaths = ['image-1440', 'image-390', 'video-1440', 'video-390']
  .map(name => path.join(screenshotDir, 'ai-media-' + name + '.png'));
const durationMs = 30000;
const identityWaitMs = 55000;
const results = { image: { status: 'not_run' }, video: { status: 'not_run' } };
const pageErrors = [];
let activeRun = null, scanConnections = 0, browser, context, page, failure = null;
const report = {
  startedAt: new Date().toISOString(),
  source: 'Synthetic repeated camera-scene.jpg via local file controls; unmodified scan WebSocket and active real AI inference',
  physicalCamera: false, websocketInterception: false, syntheticRepeatedFrames: true,
  fixture: { status: 'not_generated', path: path.relative(root, videoPath).replaceAll('\\', '/') },
  runtime: { platform: process.platform, node: process.version }, hardware: {}, mediaPermissionRequests: 0,
  sources: results, screenshots: screenshotPaths.map(file => path.relative(root, file).replaceAll('\\', '/')), pageErrors,
};

function percentile(values, p) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  return sorted.length ? sorted[Math.max(0, Math.ceil(sorted.length * p) - 1)] : null;
}
function stats(values) { return { count: values.length, p50Ms: percentile(values, .5), p95Ms: percentile(values, .95) }; }
async function readLatencyDiagnostics() {
  const text = await page.locator('.camera-diagnostics').innerText();
  const match = text.match(/AI roundtrip p50 (\d+) ms \/ p95 (\d+) ms[\s\S]*?detector \+ track p50 (\d+) ms \/ p95 (\d+) ms/);
  return match ? { aiRoundtripP50Ms: Number(match[1]), aiRoundtripP95Ms: Number(match[2]),
    detectorTrackP50Ms: Number(match[3]), detectorTrackP95Ms: Number(match[4]), source: 'scanner DOM session diagnostics; real sent-frame to returned-frame samples' }
    : { status: 'diagnostics_not_available', excerpt: text.slice(0, 500) };
}
function freshOutputs() {
  for (const file of [videoPath, reportPath, ...screenshotPaths])
    assert.equal(fs.existsSync(file), false, 'Refusing to overwrite: ' + file);
  assert.ok(fs.existsSync(imagePath), 'Source JPEG is missing');
}
function outputText(payload) {
  if (typeof payload === 'string') return payload;
  if (Buffer.isBuffer(payload)) return payload.toString('utf8');
  if (payload instanceof Uint8Array) return Buffer.from(payload).toString('utf8');
  return null;
}

async function recordMp4(bytes, crop = false) {
  console.log('Recording 30-second synthetic MP4 with installed Chrome MediaRecorder at canvas.captureStream(30).');
  const source = 'data:image/jpeg;base64,' + bytes.toString('base64');
  const recorded = await page.evaluate(async ({ source, durationMs, crop }) => {
    const mime = 'video/mp4;codecs=avc1.42001E';
    if (!MediaRecorder.isTypeSupported(mime)) throw new Error('MediaRecorder MP4/AVC unsupported');
    const image = new Image(); image.src = source;
    await Promise.race([
      new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error('Scene JPEG decode failed')); }),
      new Promise((_, reject) => setTimeout(() => reject(new Error('JPEG decode timeout')), 10000)),
    ]);
    const canvas = document.createElement('canvas'); canvas.width = crop ? 360 : image.naturalWidth; canvas.height = crop ? 480 : image.naturalHeight;
    const ctx = canvas.getContext('2d'); if (!ctx) throw new Error('Canvas context unavailable');
    const stream = canvas.captureStream(30), recorder = new MediaRecorder(stream, { mimeType: mime }), chunks = [];
    const began = performance.now(); let callbacks = 0, raf = 0;
    const stopped = new Promise((resolve, reject) => {
      recorder.ondataavailable = event => { callbacks++; if (event.data?.size) chunks.push(event.data); };
      recorder.onerror = event => reject(new Error('MediaRecorder error: ' + (event.error?.message || 'unknown')));
      recorder.onstop = resolve;
    });
    const paint = () => { if (crop) ctx.drawImage(image, 300, 120, 360, 480, 0, 0, canvas.width, canvas.height); else ctx.drawImage(image, 0, 0, canvas.width, canvas.height); raf = requestAnimationFrame(paint); };
    paint(); recorder.start(1000);
    const progress = setInterval(() => console.log('MediaRecorder still recording at ' + Math.round(performance.now() - began) + ' ms'), 5000);
    const stopTimer = setTimeout(() => { if (recorder.state !== 'inactive') recorder.stop(); }, durationMs);
    await Promise.race([stopped, new Promise((_, reject) => setTimeout(() => reject(new Error('MP4 recording exceeded 45s')), 45000))]);
    clearTimeout(stopTimer); clearInterval(progress); cancelAnimationFrame(raf); stream.getTracks().forEach(track => track.stop());
    const blob = new Blob(chunks, { type: mime }); if (!blob.size) throw new Error('Empty MP4 from MediaRecorder');
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Could not read recorded MP4')); reader.readAsDataURL(blob);
    });
    return { data, mime: recorder.mimeType, bytes: blob.size, durationMs: Math.round(performance.now() - began),
      callbacks, width: canvas.width, height: canvas.height };
  }, { source, durationMs, crop });
  const fileBytes = Buffer.from(recorded.data.slice(recorded.data.indexOf(',') + 1), 'base64');
  assert.ok(fileBytes.length > 0 && fileBytes.length <= 200 * 1024 * 1024, 'MP4 must stay within the UI 200 MiB bound');
  const fd = fs.openSync(videoPath, 'wx');
  try { fs.writeFileSync(fd, fileBytes); } finally { fs.closeSync(fd); }
  const fixture = { path: path.relative(root, videoPath).replaceAll('\\', '/'), container: 'MP4', codec: recorded.mime,
    bytes: fileBytes.length, durationMs: recorded.durationMs, targetDurationMs: durationMs, targetFps: 30,
    width: recorded.width, height: recorded.height, recorderCallbacks: recorded.callbacks,
    generation: crop ? 'Chrome MediaRecorder canvas crop x=300,y=120,w=360,h=480 from camera-scene.jpg repeated at 30 fps; synthetic, not independent footage' : 'Chrome MediaRecorder of the unchanged camera-scene.jpg repeated at 30 fps; synthetic, not independent footage',
    crop: crop ? { x: 300, y: 120, width: 360, height: 480, source: 'data/evaluation/camera-scene.jpg' } : null };
  report.fixture = { status: 'generated', ...fixture };
  console.log('MP4 ready: ' + JSON.stringify({ bytes: fixture.bytes, durationMs: fixture.durationMs, callbacks: fixture.recorderCallbacks }));
  return { bytes: fileBytes, fixture };
}

function watchRealSocket(targetPage) {
  const owner = new Map();
  targetPage.on('websocket', ws => {
    if (!new URL(ws.url()).pathname.endsWith('/api/v1/ws/scan')) return;
    scanConnections++;
    owner.set(ws, activeRun);
    const outgoing = [];
    ws.on('framesent', event => {
      if (typeof event.payload === 'string') return;
      const run = owner.get(ws); if (run) run.clientBinaryFrames++;
      outgoing.push(Date.now());
    });
    ws.on('framereceived', event => {
      const raw = outputText(event.payload); if (!raw) return;
      let data; try { data = JSON.parse(raw); } catch { return; }
      const run = owner.get(ws); if (!run) return;
      if (data.type === 'frame') {
        const sent = outgoing.shift(); run.frames.push(data);
        if (Number.isFinite(data.timing_ms?.detect_track)) run.processing.push(data.timing_ms.detect_track);
        if (Number.isFinite(sent)) run.roundtrip.push(Date.now() - sent);
      }
      if (data.type === 'identity') {
        run.identities.push(data);
        const names = JSON.stringify(data.candidates || []);
        if (data.status === 'needs_confirmation' && Number(data.votes) >= 3 && data.ocr?.number === '223'
            && names.includes('Charizard ex') && run.identityMs == null)
          run.identityMs = Date.now() - run.startedAt;
      }
    });
  });
}
async function captureWidths(name) {
  const files = [];
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1050 });
    await page.waitForTimeout(180);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true,
      'Horizontal overflow at ' + width + ' px for ' + name);
    const file = path.join(screenshotDir, 'ai-media-' + name + '-' + width + '.png');
    await page.screenshot({ path: file, fullPage: true });
    files.push(path.relative(root, file).replaceAll('\\', '/'));
  }
  await page.setViewportSize({ width: 1440, height: 1050 });
  return files;
}

function summarize(run) {
  const poseFrames = run.frames.filter(frame => typeof frame.pose_model === 'string' && frame.pose_model);
  const identity = run.identities.find(event => event.status === 'needs_confirmation' && Number(event.votes) >= 3
    && event.ocr?.number === '223' && JSON.stringify(event.candidates || []).includes('Charizard ex'));
  return {
    status: identity ? 'confirmed' : run.status || 'no_result', source: run.source,
    method: 'Local file picker; unmodified authenticated scan WebSocket; trained YOLO pose and real recognition/OCR pipeline',
    identity: identity ? { name: 'Charizard ex', set: 'Obsidian Flames', number: '223', status: identity.status,
      votes: identity.votes, consensusRequired: identity.consensus_required ?? 3 } : null,
    identityEndToEndMs: run.identityMs ?? null, frameMessages: run.frames.length,
    framesWithTrack: run.frames.filter(frame => frame.tracks?.length).length,
    clientBinaryFramesSent: run.clientBinaryFrames, clientBinaryBytes: run.binaryBytes ?? 0, poseModelFrames: poseFrames.length, poseModel: poseFrames[0]?.pose_model || null,
    processingMs: { ...stats(run.processing), includesColdStart: true }, browserSendToFrameRoundtripMs: run.latency || stats(run.roundtrip),
    actualIdentityEvents: run.identities.map(event => ({ status: event.status, votes: event.votes,
      ocrNumber: event.ocr?.number || null, candidateNames: (event.candidates || []).map(item => item.card?.name).filter(Boolean) })),
    cameraPermissionRequests: run.cameraPermissionRequests, sourceCleanup: run.cleanup || null,
    preview: run.preview || null, screenshots: run.screenshots || [], failure: run.failure || null,
  };
}

(async () => {
  try {
    freshOutputs();
    fs.mkdirSync(screenshotDir, { recursive: true });
    browser = await playwright.chromium.launch({ headless: true,
      executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
    report.runtime.chrome = browser.version();
    context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
    page = await context.newPage();
    page.on('console', message => { if (message.type() === 'log' && message.text().startsWith('MediaRecorder')) console.log(message.text()); });
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.addInitScript(() => {
      window.__aiMediaCleanup = { created: [], revoked: [], cameraPermissionRequests: 0, binaryFrameCount: 0, binaryFrameBytes: 0 };
      const create = URL.createObjectURL.bind(URL), revoke = URL.revokeObjectURL.bind(URL);
      URL.createObjectURL = value => { const url = create(value); window.__aiMediaCleanup.created.push(url); return url; };
      URL.revokeObjectURL = url => { window.__aiMediaCleanup.revoked.push(url); revoke(url); };
      const originalSend = WebSocket.prototype.send;
      WebSocket.prototype.send = function(value) {
        if (this.url?.endsWith('/api/v1/ws/scan') && typeof value !== 'string') {
          window.__aiMediaCleanup.binaryFrameCount++;
          window.__aiMediaCleanup.binaryFrameBytes += value?.size ?? value?.byteLength ?? 0;
        }
        return originalSend.call(this, value);
      };
      if (navigator.mediaDevices?.getUserMedia) {
        const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
        navigator.mediaDevices.getUserMedia = (...args) => {
          window.__aiMediaCleanup.cameraPermissionRequests++;
          return original(...args);
        };
      }
    });
    watchRealSocket(page);
    const recorded = await recordMp4(fs.readFileSync(imagePath), true);
    await page.goto('http://127.0.0.1:5173/scan');
    await page.waitForFunction(() => !document.querySelector('.camera-start')?.disabled, null, { timeout: 20000 });
    const status = await page.evaluate(async () => {
      const response = await fetch('/api/v1/live/status');
      return { code: response.status, value: await response.json() };
    });
    assert.equal(status.code, 200);
    assert.ok(status.value.pose_active, 'Trained pose model must be active');
    report.hardware = { apiCudaAvailable: status.value.cuda ?? null, apiDevice: status.value.device || 'not reported',
      activePoseModel: status.value.pose_active, bboxModels: (status.value.models || []).map(item => item.id),
      condition: 'Local Windows Chrome; API metadata reports device availability; per-process inference device not independently sampled' };
    try { report.hardware.gpu = execFileSync('nvidia-smi', ['--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
      { encoding: 'utf8', timeout: 5000 }).trim(); } catch { report.hardware.gpu = 'nvidia-smi unavailable'; }

    await page.getByRole('button', { name: 'C\u00e0i \u0111\u1eb7t camera', exact: true }).click();
    const pose = page.getByLabel('Pose keypoint quadrilateral (experimental)', { exact: true });
    assert.equal(await pose.isVisible(), true, 'Active pose option must be visible');
    if (!(await pose.isChecked())) await pose.check();
    await page.getByLabel('Hi\u1ec7n debug v\u00e0 s\u1ed1 \u0111o phi\u00ean', { exact: true }).check();

    const imageRun = { source: 'local image', frames: [], identities: [], processing: [], roundtrip: [],
      clientBinaryFrames: 0, binaryBytes: 0, startBinaryCount: 0, cameraPermissionRequests: 0, startedAt: 0 };
    imageRun.startBinaryCount = await page.evaluate(() => window.__aiMediaCleanup.binaryFrameCount);
    results.image = imageRun; activeRun = imageRun;
    await page.locator('#scan-source').selectOption('image');
    await page.locator('#scan-source-file').setInputFiles({ name: 'camera-scene.jpg', mimeType: 'image/jpeg', buffer: fs.readFileSync(imagePath) });
    imageRun.startedAt = Date.now();
    await page.getByRole('button', { name: 'Qu\u00e9t \u1ea3nh \u0111\u00e3 ch\u1ecdn', exact: true }).click();
    try { await page.locator('.recognition-status.needs_confirmation').waitFor({ timeout: identityWaitMs }); }
    catch (error) { imageRun.failure = error.message; }
    const imageConfirmed = await page.locator('.recognition-status.needs_confirmation').count() > 0;
    imageRun.cameraPermissionRequests = await page.evaluate(() => window.__aiMediaCleanup.cameraPermissionRequests);
    assert.equal(imageRun.cameraPermissionRequests, 0, 'Image must not ask for camera permission');
    if (imageConfirmed) {
      assert.equal(await page.locator('.recognition-panel>h3').innerText(), 'Charizard ex');
      const evidence = await page.locator('.recognition-evidence').innerText();
      assert.match(evidence, /223/);
      assert.ok(evidence.includes('3/'));
      assert.match(await page.locator('.camera-topline').innerText(), /STILL IMAGE.*REPEATED/);
      assert.match(await page.locator('.camera-source-note').first().innerText(), /l\u1eb7p/);
      assert.match(await page.locator('.recognition-panel').innerText(), /c\u00f9ng m\u1ed9t \u1ea3nh t\u0129nh g\u1eedi l\u1eb7p/);
      await page.locator('.tracking-quadrilaterals polygon').first().waitFor({ timeout: 5000 });
      imageRun.preview = await page.locator('.camera-topline').innerText();
      imageRun.latency = await readLatencyDiagnostics();
      imageRun.screenshots = await captureWidths('image');
      const url = await page.locator('.camera-viewport>img').getAttribute('src');
      await page.getByRole('button', { name: 'D\u1eebng qu\u00e9t', exact: true }).click();
      imageRun.cleanup = await page.evaluate(url => ({ objectUrlRevoked: window.__aiMediaCleanup.revoked.includes(url),
        mediaSourceCleared: !document.querySelector('.camera-viewport>img')?.getAttribute('src') }), url);
      assert.equal(imageRun.cleanup.objectUrlRevoked, true);
      assert.equal(imageRun.cleanup.mediaSourceCleared, true);
      const imageCounters = await page.evaluate(() => window.__aiMediaCleanup);
      imageRun.clientBinaryFrames = imageCounters.binaryFrameCount - imageRun.startBinaryCount;
      imageRun.binaryBytes = imageCounters.binaryFrameBytes;
      imageRun.status = 'confirmed';
    } else {
      imageRun.status = 'ai_failure_no_confirmation';
      imageRun.screenshots = await captureWidths('image');
    }
    results.image = summarize(imageRun);

    const videoRun = { source: 'local video', frames: [], identities: [], processing: [], roundtrip: [],
      clientBinaryFrames: 0, binaryBytes: 0, startBinaryCount: 0, cameraPermissionRequests: 0, startedAt: 0 };
    videoRun.startBinaryCount = await page.evaluate(() => window.__aiMediaCleanup.binaryFrameCount);
    results.video = videoRun; activeRun = videoRun;
    await page.locator('#scan-source').selectOption('video');
    await page.locator('#scan-source-file').setInputFiles({ name: 'ai-media-repeated-scene.mp4', mimeType: 'video/mp4', buffer: recorded.bytes });
    await page.evaluate(() => {
      window.__aiMediaVideoEnded = false; window.__aiMediaPresentedFrames = 0;
      const video = document.querySelector('video');
      video.addEventListener('ended', () => { window.__aiMediaVideoEnded = true; }, { once: true });
      if (video.requestVideoFrameCallback) {
        const next = () => { window.__aiMediaPresentedFrames++; if (!video.ended) video.requestVideoFrameCallback(next); };
        video.requestVideoFrameCallback(next);
      }
    });
    videoRun.startedAt = Date.now();
    await page.getByRole('button', { name: 'Qu\u00e9t video \u0111\u00e3 ch\u1ecdn', exact: true }).click();
    await page.waitForFunction(() => window.__aiMediaVideoEnded || document.querySelector('.tracking-box'), null, { timeout: 35000 });
    videoRun.preview = await page.locator('.camera-topline').innerText().catch(() => 'LOCAL VIDEO');
    videoRun.screenshots = await captureWidths('video');
    if (!(await page.evaluate(() => window.__aiMediaVideoEnded)))
      await page.waitForFunction(() => window.__aiMediaVideoEnded, null, { timeout: 35000 });
    const playback = await page.evaluate(() => ({ ended: window.__aiMediaVideoEnded, presentedFrames: window.__aiMediaPresentedFrames,
      srcCleared: !document.querySelector('video')?.getAttribute('src'), permissionRequests: window.__aiMediaCleanup.cameraPermissionRequests,
      revoked: window.__aiMediaCleanup.revoked.slice(), created: window.__aiMediaCleanup.created.slice(),
      binaryFrameCount: window.__aiMediaCleanup.binaryFrameCount, binaryFrameBytes: window.__aiMediaCleanup.binaryFrameBytes }));
    videoRun.cameraPermissionRequests = playback.permissionRequests;
    videoRun.clientBinaryFrames = playback.binaryFrameCount - videoRun.startBinaryCount;
    videoRun.binaryBytes = playback.binaryFrameBytes;
    videoRun.latency = await readLatencyDiagnostics();
    videoRun.cleanup = { naturalEnd: playback.ended, decodedFramesPresented: playback.presentedFrames, videoSourceCleared: playback.srcCleared,
      allObjectUrlsRevoked: playback.created.every(url => playback.revoked.includes(url)) };
    assert.equal(playback.ended, true, 'MP4 must end naturally');
    assert.ok(playback.presentedFrames > 1, 'Actual decoded MP4 frames must be presented');
    assert.equal(playback.srcCleared, true, 'Video source must clear on natural end');
    assert.equal(playback.permissionRequests, 0, 'Video must not request camera access');
    assert.ok(videoRun.cleanup.allObjectUrlsRevoked, 'Ended video must revoke its local object URL');
    assert.ok(videoRun.frames.length > 0, 'Actual video frames must reach AI');
    assert.ok(videoRun.frames.some(frame => frame.tracks?.length), 'YOLO must track a card during playback');
    assert.ok(videoRun.frames.some(frame => frame.pose_model), 'Actual video inference must use pose model');
    assert.ok(videoRun.clientBinaryFrames > 0, 'Browser must send real JPEG frames through WebSocket');
    const videoIdentity = videoRun.identities.some(event => event.status === 'needs_confirmation' && Number(event.votes) >= 3
      && event.ocr?.number === '223' && JSON.stringify(event.candidates || []).includes('Charizard ex'));
    const videoBelowConsensus = videoRun.identities.some(event => ['stabilizing', 'needs_confirmation'].includes(event.status)
      && Number(event.votes) > 0 && Number(event.votes) < 3
      && (event.ocr?.number === '223' || JSON.stringify(event.candidates || []).includes('Charizard ex')));
    const videoHasModelOutput = videoRun.frames.some(frame => frame.tracks?.length);
    videoRun.status = videoIdentity ? 'confirmed' : (videoBelowConsensus || videoHasModelOutput) ? 'no_result_within_video_duration' : 'ai_failure_no_confirmation';
    videoRun.playback = { durationSeconds: recorded.fixture.durationMs / 1000,
      decodedFramesPresented: playback.presentedFrames, naturalEnd: playback.ended };
    results.video = summarize(videoRun);
    results.video.playback = videoRun.playback;

    assert.ok(scanConnections >= 2, 'Both file types must use a real scan WebSocket');
    assert.equal(pageErrors.length, 0, 'No page runtime errors');
    assert.equal(await page.evaluate(() => window.__aiMediaCleanup.cameraPermissionRequests), 0);
    report.verdict = imageRun.status === 'confirmed' && videoRun.status === 'confirmed' ? 'passed'
      : imageRun.status === 'confirmed' && videoRun.status === 'no_result_within_video_duration'
        ? 'partial_video_no_consensus_within_duration' : 'failed';
  } catch (error) {
    failure = error;
    report.verdict = 'failed'; report.failure = error.stack || error.message;
    console.error('Verification failed: ' + error.message);
  } finally {
    if (page && !page.isClosed()) {
      try { if (await page.locator('.stop-camera:visible').count()) await page.locator('.stop-camera:visible').click({ timeout: 2000 }); } catch {}
      try { report.mediaPermissionRequests = await page.evaluate(() => window.__aiMediaCleanup?.cameraPermissionRequests ?? 0); } catch {}
    }
    report.websocketScanConnections = scanConnections;
    report.pageErrors = pageErrors;
    report.sources = { image: results.image.frames ? summarize(results.image) : results.image,
      video: results.video.frames ? summarize(results.video) : results.video };
    try {
      fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), { flag: 'wx' });
      console.log('Verification report saved: ' + path.relative(root, reportPath));
      console.log(JSON.stringify({ verdict: report.verdict, image: report.sources.image.status,
        video: report.sources.video.status, report: path.relative(root, reportPath) }));
    } catch (error) { console.error('Could not write report safely: ' + error.message); failure = failure || error; }
    if (context) await context.close().catch(() => {});
    if (browser) await browser.close().catch(() => {});
  }
  if (failure) process.exitCode = 1;
})().catch(error => { console.error(error); process.exit(1); });
