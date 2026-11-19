const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(process.env.PLAYWRIGHT_MODULE || 'D:/Project/CardLens/node_modules/playwright'); }

// Built UI + intercepted HTTP/WebSocket fixtures only; no app processes or AI inference.
const origin = 'http://127.0.0.1:5173';
const root = path.resolve(__dirname, '../dist');
const image = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAIAAAB4NzpmAAAAJElEQVR4nGN4dmUnGYiBIm37vBSIRKPaRrWNahvVhlsbnUouAKp9uCcE68OxAAAAAElFTkSuQmCC', 'base64');
const normal = { id: 'normal', label: 'Normal' };
const reverse = { id: 'reverse-holo', label: 'Reverse Holo' };
const printing = {
  id: 'fixture-charizard', name: 'Charizard ex fixture', number: '12', set_id: 'fixture-set', set_name: 'Fixture set',
  language: 'en', rarity: 'Rare', image: '/fixture.png', source: 'UI fixture', variants: [normal, reverse], reference_price: null,
};
const corrected = { ...printing, id: 'fixture-pikachu', name: 'Pikachu fixture', number: '25', variants: [reverse] };
const candidates = [printing, corrected].map(card => ({ card_id: card.id, card, similarity: .8, variant: 'normal', evidence: { variant_verified: false } }));

(async () => {
  assert.ok(fs.existsSync(path.join(root, 'index.html')), 'Run npm run build first');
  const browser = await playwright.chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
  try {
    const page = await browser.newPage({ acceptDownloads: true, viewport: { width: 1440, height: 1000 } });
    const errors = [], prices = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    const responses = {
      '/api/v1/session': {},
      '/api/v1/me': { registered: false, display_name: null, role: 'visitor' },
      '/api/v1/live/status': { models: [{ id: 'fixture-bbox', model: 'Offline UI fixture' }], active: 'fixture-bbox' },
    };
    await page.route(origin + '/**', route => {
      const request = route.request();
      const pathname = new URL(request.url()).pathname;
      if (Object.hasOwn(responses, pathname)) return route.fulfill({ json: responses[pathname] });
      if (pathname === '/api/v1/prices/batch') {
        prices.push(...request.postDataJSON().items);
        return route.fulfill({ json: [] });
      }
      if (pathname === '/fixture.png') return route.fulfill({ body: image, contentType: 'image/png' });
      const file = pathname === '/scan' ? path.join(root, 'index.html') : path.resolve(root, '.' + pathname);
      if (file.startsWith(root + path.sep) && fs.existsSync(file) && fs.statSync(file).isFile()) return route.fulfill({ path: file });
      return route.fulfill({ status: 404 });
    });
    let sendFrame;
    await page.routeWebSocket('**/api/v1/ws/scan', ws => {
      ws.onMessage(message => {
        if (typeof message !== 'string') return;
        const request = JSON.parse(message);
        if (request.type !== 'configure') return;
        const send = payload => ws.send(JSON.stringify({ configuration_id: request.configuration_id, configuration_revision: 1, ...payload }));
        sendFrame = votes => send({
          type: 'frame', timing_ms: { detect_track: 1 }, suggested_interval_ms: 250,
          tracks: [1, 2].map(id => ({
            track_id: String(id), generation: 'offline-ui', box: [.1, .1, .8, .8], confidence: .9,
            status: votes === 3 ? 'needs_confirmation' : 'stabilizing', candidates,
            quality: { usable: true, reasons: [] }, votes, consensus_required: 3,
          })),
        });
        send({ type: 'configured', pose_model: null });
        sendFrame(2);
      });
    });

    await page.goto(origin + '/scan');
    await page.getByLabel('Nguồn quét').selectOption('image');
    await page.locator('#scan-source-file').setInputFiles({ name: 'offline-fixture.png', mimeType: 'image/png', buffer: image });
    await page.getByRole('button', { name: 'Quét ảnh đã chọn', exact: true }).click();
    const panel = page.locator('.recognition-panel');
    const variant = panel.getByRole('combobox', { name: 'Biến thể bản in', exact: true });
    const save = panel.getByRole('button', { name: 'Xác nhận và lưu vào phiên', exact: true });
    const sell = panel.getByRole('link', { name: 'Đăng bán lá thẻ này', exact: true });
    await variant.waitFor();
    assert.deepEqual(await variant.locator('option').allTextContents(), ['Chọn biến thể / chưa xác minh', 'Normal', 'Reverse Holo']);
    assert.equal(await variant.inputValue(), '', 'do not assume the first catalog or server-supplied variant');
    assert.equal(await save.isDisabled(), true, 'two votes are not enough to save the AI result');
    await variant.selectOption(reverse.id);
    assert.equal(await save.isDisabled(), true);
    assert.deepEqual(prices, [], 'no price while the printing is ineligible');
    assert.equal(await sell.getAttribute('href'), '/sell?card=fixture-charizard');
    const quote = page.waitForResponse('**/api/v1/prices/batch');
    sendFrame(3);
    await quote;
    await panel.getByText('Chưa có giá', { exact: true }).waitFor();
    assert.deepEqual(prices, [{ card_id: printing.id, variant: reverse.id, condition: 'NM' }]);
    await save.click();
    const entries = page.locator('.session-cards > div');
    assert.match(await entries.first().innerText(), /Reverse Holo/);
    await page.getByRole('button', { name: 'Hoàn tác thao tác cuối', exact: true }).click();
    assert.equal(await entries.count(), 0);
    await save.click();

    await panel.getByRole('button', { name: '#2 Charizard ex fixture', exact: true }).click();
    assert.equal(await variant.inputValue(), '', 'another track of the same printing does not inherit the choice');
    await panel.getByRole('button', { name: '#1 Charizard ex fixture', exact: true }).click();
    assert.equal(await variant.inputValue(), '', 'switching back also starts blank');
    const requote = page.waitForResponse('**/api/v1/prices/batch');
    await variant.selectOption(reverse.id);
    await requote;
    await panel.getByRole('button', { name: /Pikachu fixture/ }).click();
    assert.equal(await variant.inputValue(), '', 'manual correction clears even a variant ID common to both printings');
    assert.equal(await sell.getAttribute('href'), '/sell?card=fixture-pikachu');
    await save.click();
    assert.match(await entries.last().innerText(), /Chưa xác minh.*Ảnh tĩnh lặp/);
    assert.deepEqual(prices, Array(2).fill({ card_id: printing.id, variant: reverse.id, condition: 'NM' }));

    const pendingDownload = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Xuất CSV', exact: true }).click();
    const csv = fs.readFileSync(await (await pendingDownload).path(), 'utf8');
    assert.deepEqual(csv.replace(/^\uFEFF/, '').split('\n'), [
      'card_id,name,set,number,status,source,evidence,variant_id,variant_label,variant_source,variant_verified',
      '"fixture-charizard","Charizard ex fixture","Fixture set","12","user_confirmed","image","still_image_repeated","reverse-holo","Reverse Holo","user_selected","false"',
      '"fixture-pikachu","Pikachu fixture","Fixture set","25","user_confirmed","image","still_image_repeated","","Chưa xác minh","","false"',
    ]);
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      assert.ok(await variant.isVisible());
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `horizontal overflow at ${width}px`);
    }
    assert.deepEqual(errors, []);
    console.log('Offline browser fixture passed: selection, printing/track reset, price gating, save/undo, CSV, desktop/mobile. No AI inference.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
