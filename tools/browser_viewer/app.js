'use strict';
const $=id=>document.getElementById(id), canvas=$('scene'), ctx=canvas.getContext('2d');
let session=null,index=0,playing=false,loading=false,yaw=-.7,pitch=.85,zoom=1,center=[0,0,0],span=10,follow=false;
let requested=0,urls=[],lastWall=0,playTime=0;
const pointers=new Map();
const retainedMap=new RetainedMapLayer();
let showMap=true,autoFit=true,mapStream=null,mapPollStarted=false;
let drawPending=false,lastDraw=0;
function draw(){
 if(drawPending)return;
 drawPending=true;
 setTimeout(()=>requestAnimationFrame(()=>{drawPending=false;lastDraw=performance.now();drawNow();}),Math.max(0,33-(performance.now()-lastDraw)));
}
function fitMap(){
 if(!retainedMap.bounds){fit();return;}
 const {low,high}=retainedMap.bounds;
 center=low.map((v,i)=>(v+high[i])/2);span=Math.max(1,...high.map((v,i)=>v-low[i]))*1.5;
 zoom=1;follow=false;$('follow').setAttribute('aria-pressed','false');draw();
}
function startMapPolling(){
 if(mapPollStarted)return;mapPollStarted=true;
 async function poll(){
  if(mapStream!==null&&!document.hidden&&showMap){
   const stream=mapStream,abort=new AbortController(),timer=setTimeout(()=>abort.abort(),2500);
   try{
    const response=await fetch(`/api/map?stream=${encodeURIComponent(stream)}&since=${retainedMap.revision}`,{signal:abort.signal,cache:'no-store'});
    if(stream!==mapStream)return;
    if(response.status===304)return;
    if(response.status===409)return;
    if(!response.ok)throw Error('Map unavailable');
    const buffer=await response.arrayBuffer();
    if(stream!==mapStream)return;
    if(decodeURIComponent(response.headers.get('X-Map-Stream'))!==stream)return;
    retainedMap.accept(buffer,Number(response.headers.get('X-Map-Revision')),Number(response.headers.get('X-Map-Voxel-Size')));
    $('map-count').textContent=`${(retainedMap.points.length/3).toLocaleString()} retained map points · ${retainedMap.voxelSize.toFixed(2)} m voxels`;
    if(autoFit&&!follow){const savedZoom=zoom;fitMap();zoom=savedZoom;}
    draw();
   }catch(e){if(stream===mapStream)$('map-count').textContent='Map update unavailable · retaining previous map';}
   finally{clearTimeout(timer);setTimeout(poll,1000);}
  }else setTimeout(poll,1000);
 }
 poll();
}
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function clock(t){return `${Math.floor(t/60)}:${(t%60).toFixed(1).padStart(4,'0')}`;}
function project(p){const v=p.map((n,i)=>n-center[i]),a=Math.cos(yaw),b=Math.sin(yaw),c=Math.cos(pitch),d=Math.sin(pitch);const x=a*v[0]-b*v[1],depth=b*v[0]+a*v[1],y=c*v[2]-d*depth;const s=Math.min(canvas.clientWidth,canvas.clientHeight)*zoom/span;return [canvas.clientWidth/2+x*s,canvas.clientHeight/2-y*s];}
function line(points,color,width=1){if(points.length<2)return;ctx.beginPath();points.forEach((p,i)=>{const xy=project(p);i?ctx.lineTo(...xy):ctx.moveTo(...xy)});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
function rotate(v,q){const [x,y,z,w]=q;const t=[2*(y*v[2]-z*v[1]),2*(z*v[0]-x*v[2]),2*(x*v[1]-y*v[0])];return [v[0]+w*t[0]+y*t[2]-z*t[1],v[1]+w*t[1]+z*t[0]-x*t[2],v[2]+w*t[2]+x*t[1]-y*t[0]];}
function drawNow(){const w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(devicePixelRatio||1,2);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);if(!session || !session.frames.length)return;
 const current=session.frames[index];if(follow)center=current.p.slice();
 const step=10**Math.floor(Math.log10(span/5)),cx=Math.round(center[0]/step)*step,cy=Math.round(center[1]/step)*step;
 for(let i=-8;i<=8;i++){line([[cx+i*step,cy-8*step,0],[cx+i*step,cy+8*step,0]],'#e4ebf1');line([[cx-8*step,cy+i*step,0],[cx+8*step,cy+i*step,0]],'#e4ebf1');}
 const stride=Math.max(1,Math.floor(session.frames.length/1800));
 line(session.frames.filter((_,i)=>i%stride===0).map(f=>f.p),'#d5e2ed',1.6);
 const trail=session.frames.slice(0,index+1).filter((_,i)=>i%stride===0).map(f=>f.p);trail.push(current.p);line(trail,'#167cf3',2.7);
 if(showMap)retainedMap.paint(ctx,w,h,dpr,center,yaw,pitch,zoom,span);
 const axis=span/zoom*.08;
 [[axis,0,0],[0,axis,0],[0,0,axis]].forEach((v,i)=>line([current.p,rotate(v,current.q).map((n,j)=>n+current.p[j])],['#e5606b','#24a781','#427ff2'][i],2.2));
 const p=project(current.p);ctx.beginPath();ctx.arc(...p,5,0,2*Math.PI);ctx.fillStyle='#132e49';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke();
 ctx.font='11px system-ui';ctx.fillStyle='#6f8799';ctx.fillText(`Grid ${step.toPrecision(1)} m · Body axes X / Y / Z`,14,h-36);
}
function fit(){if(!session || !session.frames.length)return;const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];session.frames.forEach(f=>f.p.forEach((v,i)=>{min[i]=Math.min(min[i],v);max[i]=Math.max(max[i],v)}));center=min.map((v,i)=>(v+max[i])/2);span=Math.max(1,...max.map((v,i)=>v-min[i]))*1.5;zoom=1;follow=false;$('follow').setAttribute('aria-pressed','false');draw();}
function error(message){$('error').textContent=message;$('error').hidden=!message;}
async function frame(next){next=clamp(next,0,session.frames.length-1);const generation=++requested;loading=true;
 try{const blobs=await Promise.all([0,1].map(async camera=>{const r=await fetch(`/api/image?camera=${camera}&frame=${next}`);if(!r.ok)throw Error(`Camera image unavailable (${r.status})`);return r.blob()}));
 const newUrls=blobs.map(b=>URL.createObjectURL(b));await Promise.all(newUrls.map(src=>new Promise((resolve,reject)=>{const img=new Image();img.onload=resolve;img.onerror=reject;img.src=src;})));
 if(generation!==requested){newUrls.forEach(u=>URL.revokeObjectURL(u));return;}
 $('left').src=newUrls[0];$('right').src=newUrls[1];urls.forEach(u=>URL.revokeObjectURL(u));urls=newUrls;index=next;const f=session.frames[index];
 $('timeline').value=index;$('clock').textContent=`${clock(f.t)} / ${clock(session.frames.at(-1).t)}`;$('frame').textContent=`Frame ${index+1} / ${session.frames.length}`;
 ['x','y','z'].forEach((k,i)=>$(k).textContent=f.p[i].toFixed(3));$('tracks0').textContent=f.observations[0];$('tracks1').textContent=f.observations[1];$('imu').textContent=f.imu;error('');draw();
 }catch(e){if(generation===requested){pause();error(`Playback paused: ${e.message||'image decode failed'}`);}}finally{if(generation===requested)loading=false;}}
function pause(){playing=false;$('play').textContent='Play';}
$('play').onclick=()=>{if(playing){pause();return;}if(index===session.frames.length-1)playTime=0;else playTime=session.frames[index].t;lastWall=performance.now();playing=true;$('play').textContent='Pause';};
$('timeline').oninput=()=>{pause();frame(Number($('timeline').value));};
$('fit').onclick=()=>{autoFit=false;fit();};$('map-toggle').onclick=()=>{showMap=!showMap;$('map-toggle').setAttribute('aria-pressed',String(showMap));draw();};$('fit-map').onclick=()=>{autoFit=false;fitMap();};$('top').onclick=()=>{pitch=Math.PI/2;yaw=0;draw();};$('follow').onclick=()=>{follow=!follow;$('follow').setAttribute('aria-pressed',String(follow));if(!follow)fit();draw();};
canvas.onpointerdown=e=>{autoFit=false;canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);};
canvas.onpointermove=e=>{if(!pointers.has(e.pointerId))return;const old=pointers.get(e.pointerId),before=[...pointers.values()];pointers.set(e.pointerId,[e.clientX,e.clientY]);if(pointers.size===1){yaw+=(e.clientX-old[0])*.007;pitch=clamp(pitch+(e.clientY-old[1])*.007,-1.5,Math.PI/2);}else{const after=[...pointers.values()],distance=p=>Math.hypot(p[0][0]-p[1][0],p[0][1]-p[1][1]);const d=distance(before);if(d>0)zoom=clamp(zoom*distance(after)/d,.15,25);}draw();};
canvas.onpointerup=canvas.onpointercancel=e=>pointers.delete(e.pointerId);
canvas.addEventListener('wheel',e=>{autoFit=false;e.preventDefault();zoom=clamp(zoom*Math.exp(-e.deltaY*.001),.15,25);draw();},{passive:false});
window.addEventListener('resize',draw);
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
setInterval(()=>{if(!playing||!session)return;const now=performance.now();playTime+=(now-lastWall)*.001*Number($('speed').value);lastWall=now;if(loading)return;let lo=0,hi=session.frames.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(session.frames[mid].t<=playTime)lo=mid;else hi=mid-1;}if(lo!==index)frame(lo);if(playTime>=session.frames.at(-1).t)pause();},100);
(async()=>{try{const r=await fetch('/api/session');if(!r.ok)throw Error('Session unavailable');session=await r.json();if(session.live){startLive();return;}$('title').textContent=session.name.replaceAll('_',' ');$('mode').textContent=session.mode;$('session').textContent=`${session.frames.length.toLocaleString()} stereo frames`;$('timeline').max=session.frames.length-1;fit();await frame(0);$('play').disabled=false;}catch(e){error(e.message);$('session').textContent='Unavailable';}})();

function startLive(){
 if(session.controls)startControls();
 startMapPolling();
 $('title').textContent='Online VIO';$('mode').textContent='Waiting for VIO';
 $('play').hidden=true;$('speed').parentElement.hidden=true;$('timeline').hidden=true;
 document.querySelector('footer').textContent='Online sensor-only VIO · Coloured axes show body orientation';
 let sequence=0,stream=null;
 async function poll(){
  try{
   const response=await fetch('/api/live');if(!response.ok)throw Error('Viewer server unavailable');
   const data=await response.json();
   $('mode').textContent=data.age===null?'Waiting for VIO':data.age>3?'VIO stream stopped':'Live VIO';
   $('session').textContent=data.age===null?'Waiting for incoming frames':`Last update ${data.age.toFixed(1)} s ago`;
   if(data.packet && data.sequence!==sequence){
    const f=data.packet;
    const sources=f.images.map(b=>'data:image/jpeg;base64,'+b);
    await Promise.all(sources.map(src=>new Promise((resolve,reject)=>{const img=new Image();img.onload=resolve;img.onerror=()=>reject(Error('Invalid stereo image'));img.src=src;})));
    if(stream!==f.stream){session.frames=[];stream=f.stream;mapStream=stream;retainedMap.reset(stream);autoFit=true;$('map-count').textContent='Loading accumulated map…';}
    $('map-tools').hidden=false;
    sequence=data.sequence;session.frames.push({t:f.t,p:f.p,q:f.q});if(session.frames.length>3000)session.frames.shift();
    index=session.frames.length-1;$('left').src=sources[0];$('right').src=sources[1];
    $('clock').textContent=clock(f.t);$('frame').textContent=`Frame ${f.frame_id}`;
    ['x','y','z'].forEach((k,i)=>$(k).textContent=f.p[i].toFixed(3));
    $('tracks0').textContent=f.observations[0];$('tracks1').textContent=f.observations[1];$('imu').textContent=f.imu;
    if(!follow&&autoFit&&!retainedMap.bounds){const savedZoom=zoom;fit();zoom=savedZoom;}draw();
   }
   error('');
  }catch(e){$('mode').textContent='Disconnected';error(e.message);}
  setTimeout(poll,150);
 }
 poll();
}

function startControls(){
 $('controls').hidden=false;
 const token=document.querySelector('meta[name="control-token"]').content;
 let pending=false;
 async function refresh(){
  try{
   const response=await fetch('/api/control');const state=await response.json();
   if(!response.ok)throw Error(state.error||'Unable to read VIO status');
   const busy=pending||!!state.operation;
   $('start-vio').disabled=busy||state.vio||state.gaussian;$('stop-vio').disabled=busy||!state.vio;
   $('vio-state').textContent=state.vio?'VIO running':state.gaussian?'VIO in Gaussian pipeline':'VIO stopped';
   $('gaussian-controls').hidden=!state.gaussian_available;
   $('start-gaussian').disabled=busy||state.vio||state.gaussian;
   $('stop-gaussian').disabled=busy||!state.gaussian;
   $('gaussian-state').textContent=state.operation?.includes('gaussian')?(state.operation==='start-gaussian'?'Starting…':'Saving map and stopping…'):state.gaussian?'Running':'Stopped';
   const trackingWarning=state.gaussian&&!state.operation?(state.mapping_tracking_lost?'Mapping paused: VIO tracking was lost. Stop and restart in a textured scene.':state.mapping_tracking_ok===false?'Mapping paused: waiting for reliable stereo tracking.':''):'';
   $('control-status').textContent=state.error||trackingWarning||state.message||'';
   $('control-status').className=(state.error||trackingWarning)?'control-error':'';
  }catch(e){$('control-status').textContent=e.message;['start-vio','stop-vio','start-gaussian','stop-gaussian'].forEach(k=>$(k).disabled=true);}
 }
 async function action(name){
  pending=true;['start-vio','stop-vio','start-gaussian','stop-gaussian'].forEach(k=>$(k).disabled=true);
  try{
   const response=await fetch('/api/'+name,{method:'POST',headers:{'X-Control-Token':token}});
   const result=await response.json();if(!response.ok)throw Error(result.error);
   pending=false;await refresh();
  }catch(e){pending=false;$('control-status').textContent=e.message;}
 }
 $('start-vio').onclick=()=>action('start');$('stop-vio').onclick=()=>action('stop');
 $('start-gaussian').onclick=()=>action('start-gaussian');$('stop-gaussian').onclick=()=>action('stop-gaussian');
 $('open-gaussian').onclick=()=>{const url=new URL(location.href);url.port='8092';url.pathname='/splats';url.search='';url.hash='live';window.open(url.href,'_blank','noopener');};
 refresh();setInterval(refresh,2000);
}
