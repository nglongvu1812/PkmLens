const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(process.env.PLAYWRIGHT_MODULE || 'D:/Project/CardLens/node_modules/playwright'); }

// UI contract fixtures only. Run browser-camera.cjs separately for actual YOLO inference.
// Default: existing 5173 + core services/auth. CAMERA_MEDIA_OFFLINE=1: intercept HTTP using dist after npm run build.
// Optional CAMERA_MEDIA_SCREENSHOT_DIR saves desktop/mobile viewport captures outside source files.
(async () => {
  const browser = await playwright.chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  });
  const errors = [], uploads = [];
  const configurationRequests = [];
  let wire, configuration, routedSockets = 0, frameCount = 0, revision = 0, suspended = false, autoFrames = true, rejectNext = false;
  let fixtureCard;
  const candidate = () => ({ card_id: fixtureCard.id, card: { ...fixtureCard, name: 'UI contract fixture' }, similarity: .812 });
  const track = (overrides = {}) => ({
    track_id: 'fixture-1', generation: 'ui-only', box: [.15, .15, .7, .8],
    quadrilateral: [[.17, .15], [.7, .2], [.67, .8], [.15, .74]],
    confidence: .91, status: 'needs_confirmation', candidates: [candidate()],
    quality: { usable: true, reasons: [] }, votes: 5, consensus_required: 5, ...overrides,
  });
  const send = (payload, config = configuration) => wire.send(JSON.stringify({
    configuration_id: config.configuration_id, configuration_revision: config.revision, ...payload,
  }));
  const sendFrame = (overrides = {}, config = configuration) => send({
    type: 'frame', tracks: [track(overrides)], timing_ms: { detect_track: 12 }, suggested_interval_ms: 150,
  }, config);
  const offline = process.env.CAMERA_MEDIA_OFFLINE === '1';
  const context = await browser.newContext({ permissions: ['camera'], viewport: { width: 1440, height: 1100 } });
  try {
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (['POST', 'PUT', 'PATCH'].includes(request.method()) && !/\/(session|prices\/batch|scan\/record)$/.test(new URL(request.url()).pathname))
        uploads.push(new URL(request.url()).pathname);
    });
    await page.addInitScript(() => {
      window.__mediaCleanup = { created: [], revoked: [], streams: [], sockets: [], permissions: 0, delayPermission: false, delayEncode: false };
      const originalCreate = URL.createObjectURL.bind(URL), originalRevoke = URL.revokeObjectURL.bind(URL);
      URL.createObjectURL = blob => { const url = originalCreate(blob); window.__mediaCleanup.created.push(url); return url; };
      URL.revokeObjectURL = url => { window.__mediaCleanup.revoked.push(url); originalRevoke(url); };
      const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      navigator.mediaDevices.getUserMedia = async constraints => {
        window.__mediaCleanup.permissions++;
        const stream = await getUserMedia(constraints); window.__mediaCleanup.streams.push(stream);
        if (window.__mediaCleanup.delayPermission) await new Promise(resolve => { window.__releasePermission = resolve; });
        return stream;
      };
      const originalBlob = HTMLCanvasElement.prototype.toBlob;
      HTMLCanvasElement.prototype.toBlob = function(callback, ...args) {
        originalBlob.call(this, blob => {
          if (window.__mediaCleanup.delayEncode) { window.__mediaCleanup.delayEncode = false; window.__releaseEncode = () => callback(blob); }
          else callback(blob);
        }, ...args);
      };
    });
    if (offline) {
      fixtureCard = { id: 'ui-contract-only', name: 'UI fixture card', number: '1', set_id: 'ui', set_name: 'UI contract fixture', language: 'en', rarity: '', image: '/fixture-card.jpg', source: 'UI test fixture', variants: [{ id: 'normal', label: 'Normal' }], reference_price: null };
      await page.route('http://127.0.0.1:5173/**', async route => {
        const url = new URL(route.request().url());
        if (url.pathname.startsWith('/api/v1/')) {
          const endpoint = url.pathname.slice('/api/v1'.length);
          const data = endpoint === '/session' ? {} : endpoint === '/me' ? { registered: false, display_name: null, role: 'visitor' }
            : endpoint === '/live/status' ? { active: 'ui-fixture-bbox', models: [{ id: 'ui-fixture-bbox', model: 'UI fixture bbox', real_photo_validated: false }], pose_active: null, pose_models: [] }
            : endpoint === '/prices/batch' ? [] : endpoint.startsWith('/shop/cards/') ? { card: fixtureCard, listings: [] } : { items: [fixtureCard] };
          await route.fulfill({ json: data }); return;
        }
        if (url.pathname === '/fixture-card.jpg') { await route.fulfill({ path: path.resolve('data/evaluation/camera-scene.jpg') }); return; }
        const resolved = path.resolve('dist', '.' + url.pathname);
        const root = path.resolve('dist') + path.sep;
        const target = resolved.startsWith(root) && fs.existsSync(resolved) && fs.statSync(resolved).isFile() ? resolved : path.resolve('dist/index.html');
        const contentType = target.endsWith('.js') ? 'application/javascript' : target.endsWith('.css') ? 'text/css' : target.endsWith('.woff2') ? 'font/woff2' : 'text/html';
        await route.fulfill({ path: target, contentType });
      });
    }
    await page.routeWebSocket('**/api/v1/ws/scan', socket => {
      wire = socket; suspended = false; routedSockets++;
      socket.onMessage(message => {
        if (typeof message === 'string') {
          const request = JSON.parse(message);
          if (request.type === 'pause') suspended = true;
          if (request.type === 'resume') suspended = false;
          if (request.type !== 'configure') return;
          const previous = configuration;
          configuration = { ...request, revision: ++revision };
          configurationRequests.push(configuration);
          if (rejectNext) {
            rejectNext = false;
            send({ type: 'error', detail: 'invalid_config' });
            return;
          }
          // Deliver stale old-revision work while the new ack is pending.
          if (previous) sendFrame({ candidates: [{ ...candidate(), card: { ...fixtureCard, name: 'STALE RESULT' } }] }, previous);
          const accepted = configuration;
          setTimeout(() => { if (wire === socket) send({ type: 'configured', pose_model: request.use_pose ? request.pose_model : null }, accepted); }, 90);
        } else {
          frameCount++;
          if (!suspended && autoFrames) { const sentConfiguration = configuration; setTimeout(() => { if (wire === socket && autoFrames && !suspended) sendFrame({}, sentConfiguration); }, 30); }
        }
      });
    });
    if (!offline) await page.route('**/api/v1/live/status', route => route.fulfill({ json: { active: 'ui-fixture-bbox', models: [{ id: 'ui-fixture-bbox', model: 'UI fixture bbox', real_photo_validated: false }], pose_active: null, pose_models: [] } }));
    await page.goto('http://127.0.0.1:5173/scan');
    // Install after Playwright's routing shim and Vite's initial HMR connection.
    await page.evaluate(() => {
      const OriginalSocket = window.WebSocket;
      window.WebSocket = class extends OriginalSocket {
        constructor(...args) {
          super(...args);
          if (new URL(this.url).pathname === '/api/v1/ws/scan') window.__mediaCleanup.sockets.push(this);
        }
      };
    });
    await page.waitForFunction(() => !document.querySelector('.camera-start')?.disabled);
    fixtureCard = await page.evaluate(async () => (await (await fetch('/api/v1/shop/catalog?limit=1')).json()).items[0]);
    assert.ok(fixtureCard?.id, 'existing catalog supplies only display art; fixture socket supplies no AI evidence');
    assert.equal(await page.locator('#scan-source').inputValue(), 'camera');
    await page.getByRole('button', { name: 'C\u00e0i \u0111\u1eb7t camera', exact: true }).click();
    assert.equal(await page.getByText('Pose keypoint quadrilateral (experimental)', { exact: true }).count(), 0, 'no pose toggle without validated active pose');
    await page.locator('#scan-source').selectOption('image');
    await page.locator('#scan-source-file').setInputFiles({ name: 'bad.svg', mimeType: 'image/svg+xml', buffer: Buffer.from('<svg/>') });
    assert.match(await page.locator('.error-state').innerText(), /PNG\/JPEG\/WEBP/);
    await page.evaluate(() => {
      const input = document.querySelector('#scan-source-file'), transfer = new DataTransfer();
      transfer.items.add(new File([new Uint8Array(20 * 1024 * 1024 + 1)], 'too-large.png', { type: 'image/png' }));
      input.files = transfer.files; input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    assert.match(await page.locator('.error-state').innerText(), /20 MiB/);
    await page.locator('#scan-source-file').setInputFiles({ name: 'corrupt.png', mimeType: 'image/png', buffer: Buffer.from('not a decoded image') });
    await page.getByRole('button', { name: 'Qu\u00e9t \u1ea3nh \u0111\u00e3 ch\u1ecdn', exact: true }).click();
    await page.locator('.error-state').waitFor();
    assert.equal(await page.locator('.camera-viewport>img').getAttribute('src'), null);
    assert.equal(await page.evaluate(() => window.__mediaCleanup.created.every(url => window.__mediaCleanup.revoked.includes(url))), true);
    const still = { name: 'local-fixture.jpg', mimeType: 'image/jpeg', buffer: fs.readFileSync(path.resolve('data/evaluation/camera-scene.jpg')) };
    await page.locator('#scan-source-file').setInputFiles(still);
    await page.getByRole('button', { name: 'Qu\u00e9t \u1ea3nh \u0111\u00e3 ch\u1ecdn', exact: true }).click();
    await page.locator('.tracking-box').first().waitFor();
    assert.equal(await page.evaluate(() => window.__mediaCleanup.permissions), 0, 'local image must not request camera');
    assert.match(await page.locator('.camera-topline').innerText(), /STILL IMAGE.*REPEATED/);
    assert.match(await page.locator('.camera-topline').innerText(), /Preview n\/a FPS.*AI .*updates\/s/);
    assert.match(await page.locator('.recognition-evidence').innerText(), /5\/5/);
    assert.match(await page.locator('.recognition-evidence').innerText(), /0\.812/);
    assert.doesNotMatch(await page.locator('.recognition-evidence').innerText(), /81%/);
    const quad = await page.locator('.tracking-quadrilaterals polygon').evaluate(element => ({ width: getComputedStyle(element).strokeWidth, effect: getComputedStyle(element).vectorEffect }));
    assert.deepEqual(quad, { width: '2px', effect: 'non-scaling-stroke' });
    assert.equal(await page.locator('.tracking-box>i').first().evaluate(element => getComputedStyle(element).display), 'none');
    await page.locator('.tracking-box').click({ position: { x: 10, y: 10 } });
    assert.equal(await page.locator('.session-cards>div').count(), 0, 'repeated frames never save automatically');
    await page.getByRole('button', { name: 'X\u00e1c nh\u1eadn v\u00e0 l\u01b0u v\u00e0o phi\u00ean', exact: true }).click();
    assert.equal(await page.locator('.session-cards>div').count(), 1);
    await page.waitForFunction(() => document.querySelectorAll('.session-cards>div').length === 1);
    await page.getByRole('button', { name: 'Ho\u00e0n t\u00e1c thao t\u00e1c cu\u1ed1i', exact: true }).click();
    assert.equal(await page.locator('.session-cards>div').count(), 0);
    await page.getByRole('button', { name: 'X\u00e1c nh\u1eadn v\u00e0 l\u01b0u v\u00e0o phi\u00ean', exact: true }).click();
    await page.locator('.session-cards button').click();
    assert.equal(await page.locator('.session-cards>div').count(), 0);
    await page.getByRole('button', { name: 'Ho\u00e0n t\u00e1c thao t\u00e1c cu\u1ed1i', exact: true }).click();
    assert.equal(await page.locator('.session-cards>div').count(), 1);
    await page.getByLabel('Hi\u1ec7n debug v\u00e0 s\u1ed1 \u0111o phi\u00ean').check();
    await page.waitForFunction(() => /p50 \d+ ms \/ p95 \d+ ms/.test(document.querySelector('.camera-diagnostics')?.textContent));
    assert.match(await page.locator('.camera-debug-overlay').innerText(), /Detector \+ track 12\.0 ms.*AI roundtrip/);
    const ratio = await page.locator('.camera-viewport').evaluate(view => {
      const image = view.querySelector('img'), rect = view.getBoundingClientRect();
      return { media: image.naturalWidth / image.naturalHeight, view: rect.width / rect.height };
    });
    assert.ok(Math.abs(ratio.media - ratio.view) < .01, 'normalized quad aligns with undistorted image');
    autoFrames = false;
    sendFrame({ status: 'uncertain', votes: 1 });
    await page.locator('.recognition-status.uncertain').waitFor();
    assert.equal(await page.getByRole('button', { name: 'X\u00e1c nh\u1eadn v\u00e0 l\u01b0u v\u00e0o phi\u00ean', exact: true }).isDisabled(), true);
    sendFrame({ status: 'reject', quality: { usable: false, reasons: ['fixture'] } });
    await page.locator('.recognition-status.reject').waitFor();
    assert.equal(await page.getByRole('button', { name: 'X\u00e1c nh\u1eadn v\u00e0 l\u01b0u v\u00e0o phi\u00ean', exact: true }).count(), 0);
    sendFrame({ consensus_required: undefined, votes: 2 });
    await page.waitForFunction(() => /2\/3/.test(document.querySelector('.recognition-evidence')?.textContent));
    autoFrames = true;
    // An encode callback held across a configure must not send old pixels into the new revision.
    await page.evaluate(() => { window.__mediaCleanup.delayEncode = true; });
    await page.waitForFunction(() => typeof window.__releaseEncode === 'function');
    const beforeConfigureFrames = frameCount;
    const oldConfiguration = configuration;
    await page.locator('.camera-settings input[type=range]').fill('0.6');
    await page.waitForFunction(() => document.querySelectorAll('.tracking-box').length === 0);
    await page.evaluate(() => { window.__releaseEncode(); delete window.__releaseEncode; });
    assert.equal(frameCount, beforeConfigureFrames, 'old encode callback discarded');
    sendFrame({ candidates: [{ ...candidate(), card: { ...fixtureCard, name: 'STALE RESULT' } }] }, oldConfiguration);
    assert.equal(await page.getByText('STALE RESULT', { exact: true }).count(), 0);
    sendFrame({ candidates: [{ ...candidate(), card: { ...fixtureCard, name: 'STALE REVISION' } }] }, { ...configuration, revision: configuration.revision + 30 });
    send({ type: 'identity', track_id: 'fixture-1', generation: 'ui-only', status: 'needs_confirmation', candidates: [{ ...candidate(), card: { ...fixtureCard, name: 'STALE IDENTITY' } }], votes: 99 }, oldConfiguration);
    assert.equal(await page.getByText('STALE REVISION', { exact: true }).count(), 0);
    assert.equal(await page.getByText('STALE IDENTITY', { exact: true }).count(), 0);
    autoFrames = true;
    await page.locator('.tracking-box').first().waitFor();
    const currentId = configuration.configuration_id;
    assert.ok(currentId > oldConfiguration.configuration_id);
    assert.equal(configuration.use_pose, false);
    rejectNext = true;
    await page.locator('.camera-settings input[type=range]').fill('0.65');
    await page.waitForFunction(() => document.querySelectorAll('.tracking-box').length === 0);
    await page.locator('.error-state').waitFor();
    await page.locator('.camera-settings input[type=range]').fill('0.7');
    await page.locator('.tracking-box').first().waitFor();
    // Keep event diagnostics bounded under repeated configuration errors/events.
    for (let i = 0; i < 45; i++) send({ type: 'error', detail: 'invalid_config' });
    await page.waitForFunction(() => document.querySelectorAll('.camera-diagnostics li').length === 40);
    await page.locator('.camera-settings input[type=range]').fill('0.55');
    await page.locator('.tracking-box').first().waitFor();
    const summary = await page.locator('.camera-diagnostics').innerText();
    assert.doesNotMatch(summary, /data:image|base64|Bearer|local-fixture\.jpg/);
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1100 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'no horizontal overflow at ' + width);
      if (process.env.CAMERA_MEDIA_SCREENSHOT_DIR) {
        fs.mkdirSync(process.env.CAMERA_MEDIA_SCREENSHOT_DIR, { recursive: true });
        await page.locator('.camera-viewport').scrollIntoViewIfNeeded();
        await page.screenshot({ path: path.join(process.env.CAMERA_MEDIA_SCREENSHOT_DIR, 'media-' + width + '.png') });
      }
    }
    const stillUrl = await page.locator('.camera-viewport>img').getAttribute('src');
    await page.getByRole('button', { name: 'D\u1eebng qu\u00e9t', exact: true }).click();
    assert.equal(await page.evaluate(url => window.__mediaCleanup.revoked.includes(url), stillUrl), true);
    assert.equal(await page.locator('.camera-viewport>img').getAttribute('src'), null);
    // Exercise the other allowlisted image decoders with browser-generated UI fixtures.
    for (const [mime, extension] of [['image/png', 'png'], ['image/webp', 'webp']]) {
      const bytes = await page.evaluate(async mime => {
        const canvas = document.createElement('canvas'); canvas.width = 180; canvas.height = 320;
        const context = canvas.getContext('2d'); context.fillStyle = '#ead9c0'; context.fillRect(0, 0, 180, 320);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, mime));
        return Array.from(new Uint8Array(await blob.arrayBuffer()));
      }, mime);
      await page.locator('#scan-source-file').setInputFiles({ name: 'ui-only.' + extension, mimeType: mime, buffer: Buffer.from(bytes) });
      await page.getByRole('button', { name: 'Qu\u00e9t \u1ea3nh \u0111\u00e3 ch\u1ecdn', exact: true }).click();
      await page.locator('.tracking-box').first().waitFor();
      const portrait = await page.locator('.camera-viewport').boundingBox();
      assert.ok(Math.abs(portrait.width / portrait.height - 180 / 320) < .01);
      await page.getByRole('button', { name: 'D\u1eebng qu\u00e9t', exact: true }).click();
    }
    await page.locator('#scan-source').selectOption('video');
    await page.evaluate(() => {
      const input = document.querySelector('#scan-source-file'), transfer = new DataTransfer();
      const file = new File(['size-contract-fixture'], 'large.webm', { type: 'video/webm' });
      Object.defineProperty(file, 'size', { value: 200 * 1024 * 1024 + 1 });
      transfer.items.add(file); input.files = transfer.files; input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    assert.match(await page.locator('.error-state').innerText(), /200 MiB/);
    // Browser-created clips are media-lifecycle fixtures, not recognition validation.
    const videoPlayback = [];
    for (const [mime, extension] of [['video/webm', 'webm'], ['video/mp4;codecs=avc1.42001E', 'mp4']]) {
      assert.equal(await page.evaluate(mime => MediaRecorder.isTypeSupported(mime), mime), true, mime + ' recording supported in installed Chrome');
      const videoBytes = await page.evaluate(async mime => {
        const canvas = document.createElement('canvas'); canvas.width = 320; canvas.height = 180;
        const stream = canvas.captureStream(12), recorder = new MediaRecorder(stream, { mimeType: mime }), chunks = [];
        recorder.ondataavailable = event => chunks.push(event.data);
        const stopped = new Promise(resolve => { recorder.onstop = resolve; });
        recorder.start(); let count = 0;
        const draw = setInterval(() => { const context = canvas.getContext('2d'); context.fillStyle = count++ % 2 ? '#ead9c0' : '#252c29'; context.fillRect(0, 0, 320, 180); }, 80);
        await new Promise(resolve => setTimeout(resolve, 1800));
        recorder.stop(); await stopped; clearInterval(draw); stream.getTracks().forEach(track => track.stop());
        return Array.from(new Uint8Array(await new Blob(chunks, { type: mime }).arrayBuffer()));
      }, mime);
      const clip = { name: 'ui-lifecycle.' + extension, mimeType: mime.split(';')[0], buffer: Buffer.from(videoBytes) };
      await page.locator('#scan-source-file').setInputFiles(clip);
      await page.locator('video').evaluate(element => {
        window.__mediaCleanup.videoEnded = false;
        element.addEventListener('ended', () => { window.__mediaCleanup.videoEnded = true; }, { once: true });
      });
      await page.getByRole('button', { name: 'Qu\u00e9t video \u0111\u00e3 ch\u1ecdn', exact: true }).click();
      await page.locator('.tracking-box').first().waitFor();
      const videoUrl = await page.locator('video').getAttribute('src');
      await page.waitForFunction(() => /Preview \d/.test(document.querySelector('.camera-topline')?.textContent));
      const frames = await page.locator('video').evaluate(element => element.getVideoPlaybackQuality().totalVideoFrames);
      assert.ok(frames > 1, extension + ' decoded actual playback frames');
      await page.getByRole('button', { name: 'T\u1ea1m d\u1eebng qu\u00e9t', exact: true }).click();
      assert.equal(await page.locator('video').evaluate(element => element.paused), true);
      await page.getByRole('button', { name: 'Ti\u1ebfp t\u1ee5c qu\u00e9t', exact: true }).click();
      await page.waitForFunction(() => window.__mediaCleanup.videoEnded, null, { timeout: 10000 });
      await page.waitForFunction(() => !document.querySelector('video').getAttribute('src'), null, { timeout: 10000 });
      assert.equal(await page.evaluate(url => window.__mediaCleanup.revoked.includes(url), videoUrl), true);
      assert.equal(await page.locator('video').evaluate(element => element.srcObject), null);
      videoPlayback.push({ container: extension, frames, ended: true, revoked: true });
    }
    // Camera permission resolved after source switch must be stopped and must not open a scanner.
    await page.locator('#scan-source').selectOption('camera');
    await page.evaluate(() => { window.__mediaCleanup.delayPermission = true; });
    await page.getByRole('button', { name: 'B\u1eadt camera \u0111\u1ec3 b\u1eaft \u0111\u1ea7u', exact: true }).click();
    await page.waitForFunction(() => typeof window.__releasePermission === 'function');
    await page.locator('#scan-source').selectOption('image');
    const socketCount = await page.evaluate(() => window.__mediaCleanup.sockets.length);
    await page.evaluate(() => { window.__releasePermission(); window.__mediaCleanup.delayPermission = false; });
    await page.waitForFunction(() => window.__mediaCleanup.streams.every(stream => stream.getTracks().every(track => track.readyState === 'ended')));
    assert.equal(await page.evaluate(() => window.__mediaCleanup.sockets.length), socketCount);
    // Unsupported preview counters are shown as unavailable rather than target/advertised FPS.
    await page.locator('#scan-source').selectOption('camera');
    await page.locator('video').evaluate(element => { element.requestVideoFrameCallback = undefined; element.getVideoPlaybackQuality = undefined; });
    await page.getByRole('button', { name: 'B\u1eadt camera \u0111\u1ec3 b\u1eaft \u0111\u1ea7u', exact: true }).click();
    await page.locator('.tracking-box').first().waitFor();
    assert.match(await page.locator('.camera-topline').innerText(), /Preview \u2014 FPS/);
    await page.getByRole('button', { name: 'D\u1eebng camera', exact: true }).click();
    await page.waitForFunction(() => window.__mediaCleanup.streams.every(stream => stream.getTracks().every(track => track.readyState === 'ended')));
    assert.equal(await page.locator('video').evaluate(element => element.srcObject === null && !element.hasAttribute('src')), true);
    await page.locator('#scan-source').selectOption('image');
    // Route leave frees the local URL and closes the inference socket.
    await page.locator('#scan-source-file').setInputFiles(still);
    await page.getByRole('button', { name: 'Qu\u00e9t \u1ea3nh \u0111\u00e3 ch\u1ecdn', exact: true }).click();
    await page.locator('.tracking-box').first().waitFor();
    const routeUrl = await page.locator('.camera-viewport>img').getAttribute('src');
    await page.getByRole('link', { name: /Xem &/ }).click();
    await page.waitForFunction(url => window.__mediaCleanup.revoked.includes(url), routeUrl);
    assert.ok(routedSockets > 0, 'cleanup exercises scanner connections');
    assert.equal(await page.evaluate(() => window.__mediaCleanup.sockets.length), routedSockets, 'cleanup recorder covers every intercepted scanner connection');
    await page.waitForFunction(() => window.__mediaCleanup.sockets.every(socket => socket.readyState === WebSocket.CLOSED));
    assert.deepEqual(uploads, [], 'no media upload or other write endpoint');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({
      source: offline ? 'Offline built UI + intercepted HTTP/WebSocket fixtures; no AI accuracy claim' : 'UI contract fixtures over intercepted WebSocket; real local session/catalog; no AI accuracy claim',
      localImage: true, webmVideo: true, videoMp4: true, videoPlayback, bounds: true, cleanup: true, pendingPermissionFence: true,
      configurationFence: true, encodeFence: true, previewFpsMeasured: true, p50p95MeasuredFromFixtureMessages: true,
      visiblePoseStroke: true, consensusRequired: true, explicitSaveUndo: true, eventLogBound: 40,
      widths: [1440, 390], configurations: configurationRequests.length, errors,
    }));
  } finally { await context.close(); await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
