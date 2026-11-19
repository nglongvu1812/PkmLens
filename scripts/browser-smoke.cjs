const assert = require('node:assert/strict');
const path = require('node:path');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(process.env.PLAYWRIGHT_MODULE || 'D:/Project/CardLens/node_modules/playwright'); }

async function register(page, prefix) {
  await page.goto('http://127.0.0.1:5173/account');
  await page.getByRole('button', { name: 'Đăng ký ngay' }).click();
  await page.getByLabel('Tên hiển thị').fill('Kiểm thử ' + prefix);
  await page.getByLabel('Email', { exact: true }).fill(`${prefix}-${Date.now()}@example.com`);
  await page.getByLabel('Mật khẩu', { exact: true }).fill('pkmlens-test-12345');
  await page.getByRole('button', { name: 'Tạo tài khoản', exact: true }).click();
  await page.locator('.account-welcome h1').waitFor();
}

(async () => {
  const browser = await playwright.chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  });
  try {
    const seller = await browser.newContext({ viewport: { width: 390, height: 844 } });
    const page = await seller.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await register(page, 'seller');
    await page.goto('http://127.0.0.1:5173/sell?card=base1%3A4%3Aen');
    await page.waitForFunction(() => document.querySelector('select[aria-label="Chọn lá thẻ"]')?.value === 'base1:4:en');
    await page.getByLabel('Giá mong muốn (VND)').fill('250000');
    await page.locator('input[type=file]').nth(0).setInputFiles(path.resolve('data/gallery/base1-4.webp'));
    await page.locator('input[type=file]').nth(1).setInputFiles(path.resolve('data/evaluation/camera-scene.jpg'));
    await page.getByRole('button', { name: 'Đăng bán lá thẻ', exact: true }).click();
    await page.locator('.form-success').waitFor({ timeout: 20000 });
    const publication = await page.locator('.form-success').innerText();
    console.log('publish:', publication);
    if (publication.startsWith('Đã gửi tin đăng.')) {
      await page.locator('.mine-row').filter({ hasText: 'Chờ kiểm tra' }).first().waitFor();
      assert.deepEqual(errors, []);
      console.log('PASS: repeated fixture photos require moderation. Fresh photos/database are needed to repeat checkout.');
      return;
    }
    await page.locator('.mine-row .icon-button').first().click();
    const listingURL = page.url();
    const buyer = await browser.newContext({ viewport: { width: 390, height: 844 } });
    const buyerPage = await buyer.newPage();
    buyerPage.on('pageerror', error => errors.push(error.message));
    await register(buyerPage, 'buyer');
    await buyerPage.goto(listingURL);
    await buyerPage.getByRole('button', { name: 'Thêm vào giỏ hàng', exact: true }).click();
    await buyerPage.goto('http://127.0.0.1:5173/cart');
    await buyerPage.getByRole('button', { name: 'Đặt hàng mô phỏng', exact: true }).click();
    await buyerPage.getByText('Đã tạo đơn hàng mô phỏng.', { exact: true }).waitFor({ timeout: 20000 });
    const orders = await buyer.request.get('http://127.0.0.1:5173/api/v1/orders');
    const order = (await orders.json())[0];
    assert.equal(order.status, 'paid_escrow');
    const shipment = await seller.request.post(`http://127.0.0.1:5173/api/v1/orders/${order.id}/ship`, { data: { simulated_tracking: 'BROWSER-DEMO-ONLY' } });
    assert.equal(shipment.status(), 200);
    const received = await buyer.request.post(`http://127.0.0.1:5173/api/v1/orders/${order.id}/confirm-received`);
    assert.equal(received.status(), 200);
    assert.equal((await received.json()).status, 'released');
    assert.deepEqual(errors, []);
    console.log('PASS: mobile register → publish → add cart → simulated pay → ship → receive; no page errors.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
