const {chromium}=require(process.env.PLAYWRIGHT_PATH);
const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/opt/google/chrome/chrome',args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8092');
  await page.waitForFunction(()=>document.getElementById('frame').textContent==='Frame 599',null,{timeout:240000});
  await page.waitForTimeout(2500);
  await page.locator('#fit-map').click();await page.waitForTimeout(150);
  const result=await page.evaluate(()=>({frame:document.getElementById('frame').textContent,retained_points:retainedMap.points.length/3,bounds:retainedMap.bounds,voxel_size:retainedMap.voxelSize,projection_ms:retainedMap.lastProjectMs}));
  await page.screenshot({path:'work/retained_map_20260917/euroc_desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(150);
  await page.screenshot({path:'work/retained_map_20260917/euroc_mobile.png',fullPage:true});
  result.errors=errors;fs.writeFileSync('work/retained_map_20260917/real_vio.json',JSON.stringify(result,null,2));
  console.log(JSON.stringify(result,null,2));if(errors.length)throw Error('Page errors');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
