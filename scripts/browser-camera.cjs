const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const usePose = process.env.CAMERA_POSE === '1';
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(process.env.PLAYWRIGHT_MODULE || 'D:/Project/CardLens/node_modules/playwright'); }

(async () => {
  const browser = await playwright.chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    args: [
      '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      '--use-file-for-fake-video-capture=' + path.resolve(process.env.CAMERA_FIXTURE || 'data/evaluation/camera.y4m'),
    ],
  });
  try {
    const context = await browser.newContext({ permissions: ['camera'], viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    const errors = [], frames = [], identities = [];
    page.on('websocket', ws => ws.on('framereceived', event => {
      try {
        const value = JSON.parse(event.payload);
        if (value.type === 'frame') frames.push(value);
        if (value.type === 'identity') identities.push(value);
      } catch {}
    }));
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5173/scan');
    await page.waitForFunction(() => !document.querySelector('.camera-start')?.disabled);
    if (usePose) {
      await page.getByRole('button', { name: 'C\u00e0i \u0111\u1eb7t camera', exact: true }).click();
      await page.getByLabel('Pose keypoint quadrilateral (experimental)', { exact: true }).check();
    }
    const started = Date.now();
    await page.getByRole('button', { name: 'Bật camera để bắt đầu' }).click();
    await page.locator('.tracking-box').first().waitFor({ timeout: 30000 });
    await page.locator('.recognition-status.needs_confirmation').waitFor({ timeout: 60000 });
    const confirmationMs = Date.now() - started;
    if (usePose) {
      await page.locator('.tracking-quadrilaterals polygon').first().waitFor();
      assert.ok(frames.some(frame => frame.pose_model));
    }
    assert.equal(await page.locator('.recognition-panel>h3').innerText(), 'Charizard ex');
    const evidence = await page.locator('.recognition-evidence').innerText();
    assert.ok(evidence.includes('223'));
    await page.locator('.candidate-options>button').nth(1).click();
    assert.equal(await page.locator('.recognition-status').innerText(), 'Bản in do bạn chọn');
    await page.locator('.candidate-options>button').first().click();
    await page.getByLabel('Tìm bản in khác trong catalog').fill('Blastoise');
    await page.locator('.manual-matches button').first().waitFor();
    const correctedName = await page.locator('.manual-matches button b').first().innerText();
    await page.locator('.manual-matches button').first().click();
    assert.equal(await page.locator('.recognition-panel>h3').innerText(), correctedName);
    await page.locator('.candidate-options>button').first().click();
    await page.getByRole('button', { name: 'Xác nhận và lưu vào phiên' }).click();
    assert.equal(await page.locator('.session-cards>div').count(), 1);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: '.impeccable/review/live-camera' + (usePose ? '-pose' : '') + '.png', fullPage: true });
    const preview = await page.locator('.camera-topline').innerText();
    await page.getByRole('button', { name: 'Tạm dừng quét', exact: true }).click();
    await page.getByRole('button', { name: 'Tiếp tục quét', exact: true }).click();
    await page.getByRole('button', { name: 'Dừng camera', exact: true }).click();
    assert.equal(await page.locator('video').evaluate(video => video.srcObject === null), true);
    assert.deepEqual(errors, []);
    const durations = frames.map(frame => frame.timing_ms.detect_track).sort((a, b) => a - b);
    const percentile = p => durations[Math.max(0, Math.ceil(durations.length * p) - 1)];
    const report = {
      source: 'synthetic scene through fake camera; real YOLO inference', physicalCamera: false,
      fixture: process.env.CAMERA_FIXTURE || 'data/evaluation/camera.y4m',
      detector: usePose ? 'pose' : 'bbox', identity: 'Charizard ex, Obsidian Flames #223',
      verifiedThreeFrames: identities.some(event => event.votes >= 3),
      confirmationMs, frames: frames.length, preview,
      processingMs: { p50: percentile(.5), p95: percentile(.95), includesColdStart: true },
      pauseResume: true, stoppedMedia: true, errors,
    };
    fs.writeFileSync('reports/browser-camera-' + (usePose ? 'pose' : 'bbox') + '.json', JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
