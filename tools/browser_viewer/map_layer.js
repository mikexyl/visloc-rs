'use strict';
// Persistent-map transport + a cached 2D projection. Geometry stays in Float32Array.
class RetainedMapLayer {
  constructor() {
    this.points = new Float32Array();
    this.bounds = null;
    this.revision = -1;
    this.stream = null;
    this.voxelSize = .1;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.cacheKey = '';
    this.rebuilds = 0;
    this.lastProjectMs = 0;
  }
  reset(stream) {
    this.stream = stream;
    this.revision = -1;
    this.points = new Float32Array();
    this.bounds = null;
    this.cacheKey = '';
  }
  accept(buffer, revision, voxelSize) {
    if (buffer.byteLength % 12 !== 0 || buffer.byteLength > 30000 * 12) throw Error('Invalid map payload');
    // Protocol is explicitly little-endian; avoid relying on host typed-array endianness.
    const nativeLittleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
    const points = new Float32Array(buffer);
    if (!nativeLittleEndian) {
      const view = new DataView(buffer);
      for (let i = 0; i < points.length; i++) points[i] = view.getFloat32(i * 4, true);
    }
    const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < points.length; i++) {
      const value = points[i];
      if (!Number.isFinite(value)) throw Error('Nonfinite map point');
      const axis = i % 3;
      low[axis] = Math.min(low[axis], value); high[axis] = Math.max(high[axis], value);
    }
    this.points = points;
    this.bounds = points.length ? {low, high} : null;
    this.revision = revision;
    this.voxelSize = voxelSize;
    this.cacheKey = '';
  }
  paint(target, width, height, dpr, center, yaw, pitch, zoom, span) {
    const key = [this.revision, width, height, dpr, ...center, yaw, pitch, zoom, span].join(',');
    if (key !== this.cacheKey) {
      const start = performance.now();
      const w = Math.round(width * dpr), h = Math.round(height * dpr);
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
      const ctx = this.ctx;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#e69a24';
      // Camera transform is computed once, not with allocations/trig per landmark.
      const a = Math.cos(yaw), b = Math.sin(yaw), c = Math.cos(pitch), d = Math.sin(pitch);
      const scale = Math.min(width, height) * zoom / span;
      const points = this.points;
      for (let i = 0; i < points.length; i += 3) {
        const x = points[i] - center[0], y = points[i+1] - center[1], z = points[i+2] - center[2];
        const px = width / 2 + (a*x - b*y)*scale;
        const py = height / 2 - (c*z - d*(b*x + a*y))*scale;
        if (px >= 0 && px <= width && py >= 0 && py <= height) ctx.fillRect(px-1, py-1, 2, 2);
      }
      this.cacheKey = key;
      this.rebuilds++;
      this.lastProjectMs = performance.now() - start;
    }
    target.drawImage(this.canvas, 0, 0, width, height);
  }
}
if (typeof module !== 'undefined') module.exports = {RetainedMapLayer};
