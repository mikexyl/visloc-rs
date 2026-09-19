const {chromium}=require(process.env.PLAYWRIGHT_PATH);
const assert=require('node:assert/strict');const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/opt/google/chrome/chrome',args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://192.168.55.1:8090');
  await page.waitForFunction(()=>!document.getElementById('controls').hidden&&document.getElementById('resource-status').textContent.startsWith('Live'));
  await page.waitForFunction(()=>!document.getElementById('start-vio').disabled);
  assert.ok(await page.locator('#start-vio').isVisible());assert.ok(await page.locator('#stop-vio').isVisible());
  assert.ok(await page.locator('#stop-vio').isDisabled());
  assert.ok((await page.locator('#resource-metrics').textContent()).includes('Uses system RAM; no dedicated VRAM'));
  await page.screenshot({path:'work/jetson_viewer_update_20260917/desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.locator('#start-vio').isVisible());assert.ok(await page.locator('#stop-vio').isVisible());
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.screenshot({path:'work/jetson_viewer_update_20260917/mobile.png',fullPage:true});
  const resources=await(await page.request.get('http://192.168.55.1:8090/api/resources')).json();
  assert.equal(resources.samples.at(-1).errors.length,0);
  assert.deepEqual(errors,[]);
  const result={passed:true,start_visible:true,start_enabled:true,stop_visible:true,stop_enabled:false,resources:resources.samples.at(-1),errors};
  fs.writeFileSync('work/jetson_viewer_update_20260917/browser.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
