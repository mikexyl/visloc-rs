// NODE_PATH must include an installed playwright package. Run against a live viewer.
const {chromium} = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:8092';
  const output = process.argv[3] || '/tmp/visloc-resource-ui';
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROME_PATH || '/opt/google/chrome/chrome', args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(url);
    await page.waitForFunction(() => document.getElementById('resource-status').textContent.startsWith('Live'));
    await page.waitForFunction(() => [...document.querySelectorAll('.resource-value')].every(e => e.textContent !== '—'));
    const api = await (await page.request.get(`${url}/api/resources`)).json();
    assert.equal(api.scope, 'viewer-host');
    assert.ok(api.samples.length <= 120);
    assert.ok(api.samples.at(-1).gpus.length >= 1, 'Requires real GPU telemetry for this smoke test');
    const sequence = api.samples.at(-1).sequence;
    await page.waitForTimeout(2200);
    const next = await (await page.request.get(`${url}/api/resources`)).json();
    assert.ok(next.samples.at(-1).sequence > sequence);
    await page.screenshot({path:`${output}/desktop.png`,fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    await page.screenshot({path:`${output}/mobile.png`,fullPage:true});

    let fixture = structuredClone(next);
    fixture.age_seconds = 10;
    await page.route('**/api/resources', route => route.fulfill({json:fixture}));
    await page.waitForFunction(() => document.getElementById('resource-status').textContent === 'Measurements stale');
    assert.ok(await page.locator('.resource-value').evaluateAll(nodes => nodes.every(n => n.textContent === '—')));

    fixture.age_seconds = 0;
    fixture.samples.at(-1).gpus = [];
    fixture.samples.at(-1).errors = ['NVIDIA monitoring unavailable'];
    await page.waitForFunction(() => document.getElementById('resource-message').textContent.includes('NVIDIA monitoring unavailable'));
    assert.equal(await page.locator('.resource-metric').count(),4);
    assert.equal(await page.locator('.resource-metric').nth(2).locator('.resource-value').textContent(),'—');

    fixture.samples.at(-1).gpus = [{id:'jetson', name:'Jetson integrated GPU',percent:42.5,memory:null,memory_kind:'shared',temperature_c:null}];
    fixture.samples.at(-1).errors = [];
    await page.waitForFunction(() => document.getElementById('resource-metrics').textContent.includes('Uses system RAM; no dedicated VRAM'));
    fixture.samples.at(-1).gpus = [next.samples.at(-1).gpus[0], {...next.samples.at(-1).gpus[0], id:'second',name:'Second GPU'}];
    await page.waitForFunction(() => document.querySelectorAll('.resource-metric').length === 6);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));

    await page.unroute('**/api/resources');
    await page.route('**/api/resources', route => route.abort());
    await page.waitForFunction(() => document.getElementById('resource-status').textContent === 'Disconnected');
    assert.ok(await page.locator('.resource-value').evaluateAll(nodes => nodes.every(n => n.textContent === '—')));
    await page.unroute('**/api/resources');
    await page.waitForFunction(() => document.getElementById('resource-status').textContent.startsWith('Live'));
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed:true, tests:['live host metrics','advancing samples','bounded history','desktop/mobile layouts','stale data','missing GPU','Jetson shared memory','multiple GPUs','disconnect/reconnect'],latest:next.samples.at(-1)},null,2));
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode=1;});
