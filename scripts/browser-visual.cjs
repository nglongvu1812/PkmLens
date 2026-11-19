const assert = require('node:assert/strict');
const fs = require('node:fs');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(process.env.PLAYWRIGHT_MODULE || 'D:/Project/CardLens/node_modules/playwright'); }

(async () => {
  fs.mkdirSync('.impeccable/review', { recursive: true });
  const browser = await playwright.chromium.launch({
    headless: true, executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5173/');
    await page.locator('.card-tile').first().waitFor();
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all([...document.images].map(image => {
        image.loading = 'eager';
        return image.decode().catch(() => {});
      }));
    });
    const missing = await page.locator('.tile-art img').evaluateAll(images => images.filter(image => !image.complete || !image.naturalWidth).map(image => image.src));
    assert.deepEqual(missing, []);
    await page.screenshot({ path: '.impeccable/review/desktop.png', fullPage: true });
    await page.screenshot({ path: '.impeccable/review/desktop-viewport.png' });
    for (const width of [320, 390, 768, 1024, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `Overflow at ${width}`);
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: '.impeccable/review/mobile.png', fullPage: true });
    await page.goto('http://127.0.0.1:5173/scan');
    await page.waitForFunction(() => !document.querySelector('.camera-start')?.disabled);
    await page.screenshot({ path: '.impeccable/review/camera-mobile.png', fullPage: true });
    await page.getByRole('button', { name: 'Cài đặt camera' }).scrollIntoViewIfNeeded();
    const controls = await page.getByRole('button', { name: 'Cài đặt camera' }).boundingBox();
    const nav = await page.locator('.mobile-nav').boundingBox();
    assert.ok(controls.y + controls.height <= nav.y, 'Camera settings covered by bottom navigation');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ imagesLoaded: true, responsiveWidths: [320, 390, 768, 1024, 1440], cameraControlsUncovered: true, errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
