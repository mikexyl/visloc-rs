'use strict';
// Independent of VIO/replay polling: resource data stays live even without frames.
(() => {
  const byId = id => document.getElementById(id);
  const panel = document.querySelector('.resources');
  const grid = byId('resource-metrics');
  const cards = new Map();
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const percentage = value => finite(value) ? `${value.toFixed(1)}%` : '—';
  const gib = bytes => (bytes / 1024 ** 3).toFixed(2);
  let lastData = null;
  let receivedAt = 0;
  let failure = '';

  function card(key, label, color) {
    if (!cards.has(key)) {
      const element = document.createElement('article');
      element.className = 'resource-metric';
      element.style.setProperty('--metric-color', color);
      element.innerHTML = '<span class="resource-label"></span><strong class="resource-value">—</strong><span class="resource-detail"></span><svg class="resource-chart" viewBox="0 0 260 64" preserveAspectRatio="none" role="img"><path class="grid" d="M0 4H260 M0 32H260 M0 60H260"/><text x="2" y="12">100%</text><path class="trace"/></svg>';
      grid.appendChild(element);
      cards.set(key, element);
    }
    const element = cards.get(key);
    element.querySelector('.resource-label').textContent = label;
    return element;
  }

  function update(key, label, value, detail, color, samples, getter, stale) {
    const element = card(key, label, color);
    element.querySelector('.resource-value').textContent = stale ? '—' : percentage(value);
    element.querySelector('.resource-detail').textContent = stale ? 'Waiting for fresh measurements' : detail;
    const chart = element.querySelector('svg');
    chart.setAttribute('aria-label', `${label}: ${stale ? 'stale measurements' : percentage(value)}. History over two minutes, scale 0 to 100 percent.`);
    const end = samples.at(-1)?.timestamp;
    let path = '', pen = false, previous = null;
    for (const sample of samples) {
      const v = getter(sample);
      if (!finite(v) || sample.timestamp < end - 120) { pen = false; continue; }
      if (previous !== null && sample.timestamp - previous > 3) pen = false;
      const x = 260 * (1 - (end - sample.timestamp) / 120);
      const y = 60 - Math.max(0, Math.min(100, v)) * .56;
      path += `${pen ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)} `;
      pen = true;
      previous = sample.timestamp;
    }
    element.querySelector('.trace').setAttribute('d', path);
  }

  function render() {
    const samples = lastData?.samples || [];
    const latest = samples.at(-1);
    const age = latest && finite(lastData.age_seconds) ? lastData.age_seconds + (performance.now() - receivedAt) / 1000 : Infinity;
    const stale = !latest || age > 3 || !!failure;
    panel.classList.toggle('is-stale', stale);
    byId('resource-status').classList.toggle('stale', stale);
    byId('resource-status').textContent = failure ? 'Disconnected' : !latest ? 'Waiting for measurements' : stale ? 'Measurements stale' : `Live · ${age.toFixed(0)} s ago`;
    if (lastData) byId('resource-host').textContent = `Viewer host · ${lastData.hostname} · ${lastData.logical_cpus || '?'} logical CPUs`;
    byId('resource-message').textContent = failure || (latest?.errors || []).join(' · ');
    const keys = new Set(['cpu', 'ram']);
    update('cpu', 'CPU', latest?.cpu_percent, 'Across all CPU cores', '#167cf3', samples, s => s.cpu_percent, stale);
    const memoryText = memory => memory ? `${gib(memory.used_bytes)} / ${gib(memory.total_bytes)} GiB` : 'Unavailable';
    update('ram', 'RAM', latest?.memory?.percent, memoryText(latest?.memory), '#17a88f', samples, s => s.memory?.percent, stale);
    for (const gpu of latest?.gpus || []) {
      const key = `gpu:${gpu.id}`, vramKey = `vram:${gpu.id}`;
      keys.add(key); keys.add(vramKey);
      const getGpu = s => s.gpus.find(g => g.id === gpu.id);
      update(key, 'GPU', gpu.percent, `${gpu.name}${finite(gpu.temperature_c) ? ` · ${gpu.temperature_c.toFixed(0)} °C` : ''}`, '#8c62d2', samples, s => getGpu(s)?.percent, stale);
      if (gpu.memory_kind === 'shared') {
        update(vramKey, 'GPU memory · shared', null, 'Uses system RAM; no dedicated VRAM', '#de9226', samples, () => null, stale);
      } else {
        update(vramKey, 'VRAM', gpu.memory?.percent, `${memoryText(gpu.memory)} · ${gpu.name}`, '#de9226', samples, s => getGpu(s)?.memory?.percent, stale);
      }
    }
    if (!latest?.gpus?.length) {
      keys.add('gpu-unavailable'); keys.add('vram-unavailable');
      update('gpu-unavailable', 'GPU', null, 'Measurements unavailable', '#8c62d2', samples, () => null, stale);
      update('vram-unavailable', 'VRAM', null, 'Measurements unavailable', '#de9226', samples, () => null, stale);
    }
    for (const [key, element] of cards) {
      if (!keys.has(key)) { element.remove(); cards.delete(key); }
    }
  }

  async function poll() {
    if (!document.hidden) {
      const abort = new AbortController();
      const timer = setTimeout(() => abort.abort(), 2500);
      try {
        const response = await fetch('/api/resources', {cache: 'no-store', signal: abort.signal});
        if (!response.ok) throw Error(`Resource monitoring unavailable (${response.status})`);
        const data = await response.json();
        if (!Array.isArray(data.samples)) throw Error('Invalid resource measurements');
        lastData = data;
        receivedAt = performance.now();
        failure = '';
      } catch (error) {
        failure = error.name === 'AbortError' ? 'Resource monitoring timed out' : error.message;
      } finally { clearTimeout(timer); }
      render();
    }
    setTimeout(poll, 1000);
  }
  document.addEventListener('visibilitychange', render);
  render();
  poll();
})();
