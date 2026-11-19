const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(process.env.PLAYWRIGHT_MODULE || 'D:/Project/CardLens/node_modules/playwright'); }

// Offline built UI. Every application API, image, font, and shell request is intercepted; no database writes or inference.
const origin = 'http://127.0.0.1:5173';
const root = path.resolve(__dirname, '../dist');
const screenshots = path.join(process.env.TEMP || 'C:/Users/Admin/AppData/Local/Temp/opencode', 'pkmlens-browser-catalog-scale');
const image = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 250"><rect width="180" height="250" rx="12" fill="#eee2ce"/><rect x="12" y="12" width="156" height="226" rx="9" fill="#fffaf2" stroke="#c58a52" stroke-width="4"/><circle cx="90" cy="82" r="38" fill="#f0ca72"/><path d="m64 91 18-23 14 14 17-18 16 29-18 14H81z" fill="#cc6432" stroke="#292b2e" stroke-width="3"/><path d="M27 167h126M37 184h106M48 201h84" stroke="#ad987c" stroke-linecap="round" stroke-width="6"/></svg>');
const variants = [{ id: 'normal', label: 'Normal' }, { id: 'reverse-holo', label: 'Reverse Holo' }];
const english = {
  id: 'fixture-en', name: 'Pikachu English', number: '25', set_id: 'fixture-set', set_name: 'Fixture set',
  language: 'en', rarity: 'Rare', image: '/fixture.png', source: 'UI fixture', variants, reference_price: null,
};
const japanese = {
  id: 'fixture-ja', name: 'Pikachu Japanese', number: '25', set_id: 'fixture-ja-set', set_name: 'Japanese fixture',
  language: 'ja', rarity: 'Rare', image: '/fixture.png', source: 'UI fixture', variants: [variants[0]], reference_price: null,
};
const otherLanguage = {
  id: 'fixture-other', name: 'Pikachu rare locale', number: '25', set_id: 'fixture-other-set', set_name: 'Other fixture',
  language: 'xx-rare', rarity: 'Rare', image: '/fixture.png', source: 'UI fixture', variants: [variants[0]], reference_price: null,
};
const cards = [english, japanese, otherLanguage];

(async () => {
  assert.ok(fs.existsSync(path.join(root, 'index.html')), 'Run npm run build first');
  fs.mkdirSync(screenshots, { recursive: true });
  const browser = await playwright.chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [], catalogRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    const sets = [
      { id: 'fixture-set', name: 'Fixture set', count: 25, language: 'en' },
      { id: 'fixture-ja-set', name: 'Japanese fixture', count: 1, language: 'ja' },
      { id: 'fixture-other-set', name: 'Other fixture', count: 1, language: 'xx-rare' },
    ];
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.origin !== origin) return route.abort();
      if (url.pathname === '/api/v1/session') return route.fulfill({ json: {} });
      if (url.pathname === '/api/v1/me') return route.fulfill({ json: { registered: true, display_name: 'UI fixture seller', role: 'collector' } });
      if (url.pathname === '/api/v1/shop/sets') return route.fulfill({ json: sets });
      if (url.pathname === '/api/v1/shop/catalog') {
        const request = { language: url.searchParams.get('language'), offset: url.searchParams.get('offset'), set_id: url.searchParams.get('set_id') };
        catalogRequests.push(request);
        const filtered = cards.filter(card => (!request.language || card.language === request.language) && (!request.set_id || card.set_id === request.set_id));
        const offset = Number(request.offset || 0);
        const total = request.language === 'ja' ? 1 : request.language === 'xx-rare' ? 1 : request.language === 'en' ? 25 : 25;
        return route.fulfill({ json: { items: filtered.slice(offset, offset + 24), total, offset, limit: 24 } });
      }
      if (url.pathname.startsWith('/api/v1/shop/cards/')) {
        const cardId = decodeURIComponent(url.pathname.slice('/api/v1/shop/cards/'.length));
        const card = cards.find(value => value.id === cardId);
        return route.fulfill(card ? { json: { card, listings: [] } } : { status: 404, json: { detail: 'card_not_found' } });
      }
      if (url.pathname === '/api/v1/shop/mine') return route.fulfill({ json: [] });
      if (url.pathname === '/api/v1/prices/batch') {
        return route.fulfill({ json: requestJson(route.request()).items.map(item => ({ amount_vnd: null, variant: item.variant, condition: item.condition, reason: null, source: null, source_at: null })) });
      }
      if (url.pathname.endsWith('/image') || url.pathname === '/fixture.png') return route.fulfill({ body: image, contentType: 'image/svg+xml' });
      const file = path.resolve(root, '.' + decodeURIComponent(url.pathname));
      if (file.startsWith(root + path.sep) && fs.existsSync(file) && fs.statSync(file).isFile()) return route.fulfill({ path: file });
      return route.fulfill({ path: path.join(root, 'index.html') });
    });

    const waitForCatalog = predicate => page.waitForResponse(response => {
      if (!response.url().startsWith(origin + '/api/v1/shop/catalog')) return false;
      return predicate(new URL(response.url()).searchParams);
    });
    await page.goto(origin + '/');
    const languageSelect = page.locator('.filter-sidebar').getByLabel('Ngôn ngữ', { exact: true });
    await languageSelect.waitFor();
    await page.locator('.filter-sidebar select[aria-label="Ngôn ngữ"] option[value="xx-rare"]').waitFor({ state: 'attached' });
    assert.deepEqual(await languageSelect.locator('option').evaluateAll(options => options.map(option => [option.value, option.textContent])), [
      ['', 'Tất cả ngôn ngữ'], ['en', 'EN · English'], ['ja', 'JA · 日本語'], ['zh-Hans', 'zh-Hans · 简体中文'], ['xx-rare', 'xx-rare'],
    ]);
    const setOptions = await page.locator('.filter-sidebar .filter-label select').first().locator('option').evaluateAll(options => options.map(option => [option.value, option.textContent]));
    const setIds = setOptions.map(([id]) => id);
    assert.deepEqual(setIds, ['', 'fixture-set', 'fixture-ja-set', 'fixture-other-set']);
    assert.equal(new Set(setIds.slice(1)).size, setIds.length - 1, 'each set appears only once');
    assert.match(setOptions[2][1], /JA/, 'set option shows its language');
    assert.match(setOptions[3][1], /xx-rare/, 'set option falls back to the raw language code');
    const nextPage = waitForCatalog(params => params.get('offset') === '24');
    await page.getByRole('button', { name: 'Trang tiếp', exact: true }).click();
    await nextPage;
    const japaneseRequest = waitForCatalog(params => params.get('language') === 'ja');
    await languageSelect.selectOption('ja');
    const response = await japaneseRequest;
    assert.equal(new URL(response.url()).searchParams.get('offset'), '0', 'language changes restart pagination');
    await page.waitForURL('**/?lang=ja');
    const clearLanguage = page.getByRole('button', { name: /JA · 日本語/ });
    const clearedRequest = waitForCatalog(params => !params.has('language'));
    await clearLanguage.click();
    const cleared = await clearedRequest;
    assert.equal(new URL(cleared.url()).searchParams.get('offset'), '0');
    await page.waitForURL('http://127.0.0.1:5173/');
    assert.equal(await languageSelect.inputValue(), '', 'clearing the chip clears the language selection');
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `market has no horizontal overflow at ${width}px`);
      if (width === 390) {
        await page.getByRole('button', { name: 'Mở bộ lọc' }).click();
        const mobileLanguage = page.locator('.filter-dialog').getByLabel('Ngôn ngữ', { exact: true });
        const mobileRequest = waitForCatalog(params => params.get('language') === 'ja');
        await mobileLanguage.selectOption('ja');
        await mobileRequest;
        await page.locator('.filter-dialog').getByRole('button', { name: /Xem k.t qu./ }).click();
        await page.waitForURL('**/?lang=ja');
        const mobileClearRequest = waitForCatalog(params => !params.has('language'));
        await page.getByRole('button', { name: /JA · 日本語/ }).click();
        await mobileClearRequest;
        await page.waitForURL('http://127.0.0.1:5173/');
        assert.equal(await languageSelect.inputValue(), '', 'mobile language chip clears the language filter');
      }
      await page.screenshot({ path: path.join(screenshots, `market-${width}.png`) });
    }

    await page.goto(origin + '/cards/fixture-ja');
    await page.getByRole('heading', { name: 'Pikachu Japanese', exact: true }).waitFor();
    assert.equal(await page.locator('.detail-tags .tag.neutral').first().textContent(), 'JA · 日本語', 'detail identifies Japanese card language');
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Japanese detail has no horizontal overflow at ${width}px`);
      await page.screenshot({ path: path.join(screenshots, `detail-ja-${width}.png`), fullPage: true });
    }
    await page.goto(origin + '/cards/fixture-other');
    await page.getByRole('heading', { name: 'Pikachu rare locale', exact: true }).waitFor();
    assert.equal(await page.locator('.detail-tags .tag.neutral').first().textContent(), 'xx-rare', 'unknown language codes remain unchanged');

    await page.goto(origin + '/sell');
    const variantSelect = page.getByLabel('Biến thể', { exact: true });
    await variantSelect.waitFor();
    assert.equal(await variantSelect.inputValue(), '', 'no card and no variant query does not select a variant');
    assert.equal(await page.getByRole('button', { name: /Đăng bán lá thẻ/ }).isDisabled(), true, 'selling stays disabled without a variant');
    for (const [url, expected] of [
      ['', ''],
      ['&variant=not-a-real-variant', ''],
      ['&variant=reverse-holo', 'reverse-holo'],
    ]) {
      await page.goto(origin + '/sell?card=fixture-en' + url);
      await page.waitForFunction(() => document.querySelector("#sell-card")?.value === "fixture-en" && document.querySelector("#sell-variant")?.options.length === 3);
      assert.equal(await page.locator("#sell-card").inputValue(), "fixture-en", "the query-selected card loads");
      assert.equal(await variantSelect.inputValue(), expected, `${url} preselection is validated against the card variants`);
      assert.equal(await page.getByRole('button', { name: /Đăng bán lá thẻ/ }).isDisabled(), !expected, 'invalid or missing variant keeps publishing disabled');
      for (const width of [1440, 390]) {
        await page.setViewportSize({ width, height: 1000 });
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `sell has no horizontal overflow at ${width}px`);
        await page.screenshot({ path: path.join(screenshots, `sell-${expected || 'blank'}-${width}.png`), fullPage: true });
      }
    }
    assert.ok(catalogRequests.some(request => request.language === 'ja' && request.offset === '0'), 'Japanese catalog filter reached the API and reset offset');
    assert.deepEqual(errors, [], 'browser has no JavaScript or console errors');
    console.log(`Offline catalog UI checks passed: language labels/filter and reset, Detail locale, Sell variant validation, 390/1440 screenshots (${screenshots}). No database writes or inference.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

function requestJson(request) { return request.postDataJSON(); }
