// Requires Playwright and a freshly started local live_server.py.
const {chromium} = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
(async()=>{
 const url=process.argv[2]||'http://127.0.0.1:8092',out=process.argv[3]||'/tmp/visloc-map-ui';
 fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'/opt/google/chrome/chrome',args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));await page.goto(url);
  const image=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=c.height=32;return c.toDataURL('image/jpeg').split(',')[1];});
  let stream='map-test-'+Date.now(),frame=0;
  async function publish(points){const response=await page.request.post(url+'/api/live',{data:{stream,frame_id:frame++,t:frame*.05,p:[0,0,0],q:[0,0,0,1],observations:[20,20],imu:10,images:[image,image],map_points:points}});assert.equal(response.status(),200);}
  async function waitCount(n){await page.waitForFunction(n=>retainedMap.points.length===n*3,n);}
  await publish([[1,0,0,0],[2,1,0,0]]);await waitCount(2);
  await publish([[3,10,0,0],[4,11,0,0]]);await waitCount(4);
  await publish([]);await page.waitForTimeout(1300);await waitCount(4);
  await publish([[1,2,0,0]]);
  await page.waitForFunction(()=>Array.from(retainedMap.points).includes(2));
  assert.deepEqual((await page.evaluate(()=>Array.from(retainedMap.points).filter((_,i)=>i%3===0))).sort((a,b)=>a-b),[1,2,10,11]);
  const live=await (await page.request.get(url+'/api/live')).json();assert.equal(live.packet.map_points,undefined);
  const metadata=await page.evaluate(()=>({revision:retainedMap.revision,stream:mapStream}));
  const unchanged=await page.request.get(`${url}/api/map?stream=${metadata.stream}&since=${metadata.revision}`);assert.equal(unchanged.status(),304);
  assert.equal((await page.request.get(url+'/api/map?stream=wrong')).status(),409);
  await page.reload();await waitCount(4); // Joining late still receives the whole retained map.
  stream+='-capacity';frame=0;
  for(let start=0;start<30000;start+=3000){await publish(Array.from({length:3000},(_,j)=>{const i=start+j;return[i,(i%200)*.2,(Math.floor(i/200)%150)*.2,Math.sin(i*.02)];}));}
  await waitCount(30000);await page.locator('#fit-map').click();await page.waitForTimeout(250);
  const stats=await page.evaluate(()=>{
   const started=performance.now();
   retainedMap.cacheKey='';drawNow();const rebuild=retainedMap.lastProjectMs;
   const count=retainedMap.rebuilds;for(let i=0;i<20;i++)drawNow();
   return {count:retainedMap.points.length/3,geometryBytes:retainedMap.points.byteLength,projectionMs:rebuild,cachedRebuilds:retainedMap.rebuilds-count,draw20Ms:performance.now()-started};
  });
  assert.equal(stats.cachedRebuilds,0);
  assert.equal(stats.geometryBytes,360000);
  await page.screenshot({path:out+'/desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(200);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.screenshot({path:out+'/mobile.png',fullPage:true});
  // Run past the cap: adaptive coarsening keeps coverage of both old and new areas.
  await publish(Array.from({length:3000},(_,j)=>[30000+j,80+(j%100)*.2,Math.floor(j/100)*.2,0]));
  await page.waitForFunction(()=>retainedMap.voxelSize>.1);
  const bounds=await page.evaluate(()=>({count:retainedMap.points.length/3,bounds:retainedMap.bounds,voxel:retainedMap.voxelSize}));
  assert.ok(bounds.count<=30000);assert.ok(bounds.bounds.low[0]<1&&bounds.bounds.high[0]>80);
  stream+='-reset';await publish([[1,0,0,0]]);await waitCount(1);
  assert.deepEqual(errors,[]);
  const result={passed:true,tests:['retired points retained','position correction','empty window','cached binary response','stream isolation','late join/reload','30000-point rendering cache','phone layout','adaptive capacity preserves old/new coverage','new-session reset'],stats,bounds};
  fs.writeFileSync(out+'/browser.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
