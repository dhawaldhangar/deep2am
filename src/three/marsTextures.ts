import * as THREE from 'three';

function hash2D(ix: number, iy: number, seed = 1337): number {
  const n = Math.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453123;
  return n - Math.floor(n);
}

function valueNoise2D(x: number, y: number, period: number, seed = 1337): number {
  const ix0 = Math.floor(x);
  const iy0 = Math.floor(y);
  const fx = x - ix0;
  const fy = y - iy0;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);

  const wx0 = ((ix0 % period) + period) % period;
  const wy0 = ((iy0 % period) + period) % period;
  const wx1 = (wx0 + 1) % period;
  const wy1 = (wy0 + 1) % period;

  const a = hash2D(wx0, wy0, seed);
  const b = hash2D(wx1, wy0, seed);
  const c = hash2D(wx0, wy1, seed);
  const d = hash2D(wx1, wy1, seed);

  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

function tileableFbm(u: number, v: number, baseFreq: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let freq = baseFreq;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * valueNoise2D(u * freq, v * freq, freq, seed + i * 31);
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return sum / norm;
}

/**
 * High-resolution (512x512) tileable Martian regolith albedo + Sobel tangent normal map + bump map.
 */
export function createMarsRegolithTextures(maxAnisotropy: number): {
  map: THREE.DataTexture;
  normalMap: THREE.DataTexture;
  bumpMap: THREE.DataTexture;
} {
  const size = 512;
  const count = size * size;
  const heightField = new Float32Array(count);
  const colorData = new Uint8Array(count * 4);
  const normalData = new Uint8Array(count * 4);
  const bumpData = new Uint8Array(count * 4);

  // 1. Deterministic multi-octave FBM + wind-rippled sand heightfield
  for (let y = 0; y < size; y++) {
    const v = y / size;
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const macro = tileableFbm(u, v, 6, 4, 101);
      const micro = tileableFbm(u, v, 24, 3, 409);
      // Wind-blown aeolian ripples warped slightly by macro noise
      const warp = macro * 0.45 + micro * 0.12;
      const ripplePhase = (u * 18.0 + v * 7.0 + warp * 2.2) * Math.PI * 2;
      const ripples = Math.sin(ripplePhase) * 0.5 + 0.5;
      const grit = hash2D(x, y, 777);

      const h = macro * 0.52 + micro * 0.24 + ripples * 0.18 + grit * 0.06;
      heightField[y * size + x] = h;

      // Blend rust-orange, warm ochre crests, and dark basalt troughs
      const crest = Math.min(1, Math.max(0, (h - 0.35) * 1.9));
      const basaltMask = Math.max(0, 1 - macro * 2.25);

      let r = 158 + crest * 54 - basaltMask * 68 + (grit - 0.5) * 16;
      let g = 74 + crest * 42 - basaltMask * 34 + (grit - 0.5) * 10;
      let b = 44 + crest * 22 - basaltMask * 16 + (grit - 0.5) * 8;

      const idx = (y * size + x) * 4;
      colorData[idx] = Math.max(0, Math.min(255, Math.round(r)));
      colorData[idx + 1] = Math.max(0, Math.min(255, Math.round(g)));
      colorData[idx + 2] = Math.max(0, Math.min(255, Math.round(b)));
      colorData[idx + 3] = 255;

      const bumpByte = Math.max(0, Math.min(255, Math.round(h * 255)));
      bumpData[idx] = bumpByte;
      bumpData[idx + 1] = bumpByte;
      bumpData[idx + 2] = bumpByte;
      bumpData[idx + 3] = 255;
    }
  }

  // 2. Compute tangent-space normal map via central differences
  const strength = 4.2;
  for (let y = 0; y < size; y++) {
    const ym = (y - 1 + size) % size;
    const yp = (y + 1) % size;
    for (let x = 0; x < size; x++) {
      const xm = (x - 1 + size) % size;
      const xp = (x + 1) % size;

      const hL = heightField[y * size + xm];
      const hR = heightField[y * size + xp];
      const hD = heightField[ym * size + x];
      const hU = heightField[yp * size + x];

      const dx = (hR - hL) * strength;
      const dy = (hU - hD) * strength;
      const nz = 1.0;
      const invLen = 1 / Math.hypot(dx, dy, nz);

      const nx = -dx * invLen;
      const ny = -dy * invLen;
      const nzz = nz * invLen;

      const idx = (y * size + x) * 4;
      normalData[idx] = Math.round((nx * 0.5 + 0.5) * 255);
      normalData[idx + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      normalData[idx + 2] = Math.round((nzz * 0.5 + 0.5) * 255);
      normalData[idx + 3] = 255;
    }
  }

  const map = new THREE.DataTexture(colorData, size, size, THREE.RGBAFormat);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(16, 16);
  map.anisotropy = maxAnisotropy;
  map.generateMipmaps = true;
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.needsUpdate = true;

  const normalMap = new THREE.DataTexture(normalData, size, size, THREE.RGBAFormat);
  normalMap.colorSpace = THREE.NoColorSpace;
  normalMap.wrapS = THREE.RepeatWrapping;
  normalMap.wrapT = THREE.RepeatWrapping;
  normalMap.repeat.set(16, 16);
  normalMap.anisotropy = maxAnisotropy;
  normalMap.generateMipmaps = true;
  normalMap.minFilter = THREE.LinearMipmapLinearFilter;
  normalMap.needsUpdate = true;

  const bumpMap = new THREE.DataTexture(bumpData, size, size, THREE.RGBAFormat);
  bumpMap.colorSpace = THREE.NoColorSpace;
  bumpMap.wrapS = THREE.RepeatWrapping;
  bumpMap.wrapT = THREE.RepeatWrapping;
  bumpMap.repeat.set(16, 16);
  bumpMap.anisotropy = maxAnisotropy;
  bumpMap.generateMipmaps = true;
  bumpMap.minFilter = THREE.LinearMipmapLinearFilter;
  bumpMap.needsUpdate = true;

  return { map, normalMap, bumpMap };
}

/**
 * High-res PBR solar cell array texture with sapphire silicon cells, fine silver busbars, and specular grid.
 */
export function createSolarCellTexture(maxAnisotropy: number): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  const grad = ctx.createLinearGradient(0, 0, size, size);
  grad.addColorStop(0, '#071630');
  grad.addColorStop(0.5, '#0D264D');
  grad.addColorStop(1, '#061228');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  const cols = 8;
  const rows = 5;
  const cw = size / cols;
  const ch = size / rows;

  // Individual photovoltaic wafer cells with diagonal bevels
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cw + 3;
      const y = r * ch + 3;
      const w = cw - 6;
      const h = ch - 6;
      ctx.fillStyle = (r + c) % 2 === 0 ? '#0B2347' : '#0E2A56';
      ctx.fillRect(x, y, w, h);

      // Fine horizontal collector fingers
      ctx.strokeStyle = 'rgba(148, 197, 255, 0.16)';
      ctx.lineWidth = 1;
      for (let fy = y + 8; fy < y + h - 4; fy += 8) {
        ctx.beginPath();
        ctx.moveTo(x + 2, fy);
        ctx.lineTo(x + w - 2, fy);
        ctx.stroke();
      }
    }
  }

  // Primary silver-cyan busbars
  ctx.strokeStyle = '#38BDF8';
  ctx.lineWidth = 2.5;
  for (let c = 0; c <= cols; c++) {
    ctx.beginPath();
    ctx.moveTo(c * cw, 0);
    ctx.lineTo(c * cw, size);
    ctx.stroke();
  }
  for (let r = 0; r <= rows; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * ch);
    ctx.lineTo(size, r * ch);
    ctx.stroke();
  }

  ctx.strokeStyle = 'rgba(186, 230, 253, 0.45)';
  ctx.lineWidth = 1.5;
  for (let c = 0; c < cols; c++) {
    const bx = c * cw + cw * 0.5;
    ctx.beginPath();
    ctx.moveTo(bx, 0);
    ctx.lineTo(bx, size);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = maxAnisotropy;
  return tex;
}

/**
 * Aerospace hull paneling with metallic seams, rivet accents, and a lower-edge Martian regolith dust gradient.
 */
export function createHabitatHullTexture(maxAnisotropy: number): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // Base composite white-slate finish with visible lower-edge Martian dust splash
  const vGrad = ctx.createLinearGradient(0, 0, 0, size);
  vGrad.addColorStop(0.0, '#E8EEF5');
  vGrad.addColorStop(0.42, '#CBD5E1');
  vGrad.addColorStop(0.68, '#B8836C');
  vGrad.addColorStop(0.86, '#964E32');
  vGrad.addColorStop(1.0, '#6E321D');
  ctx.fillStyle = vGrad;
  ctx.fillRect(0, 0, size, size);

  // Panel seam lines
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.52)';
  ctx.lineWidth = 2.5;
  for (let i = 1; i < 8; i++) {
    const x = (i / 8) * size;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, size);
    ctx.stroke();
  }
  for (let j = 1; j < 4; j++) {
    const y = (j / 4) * size;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(size, y);
    ctx.stroke();
  }

  // Subtle specular highlight strip along seams
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.42)';
  ctx.lineWidth = 1.2;
  for (let i = 1; i < 8; i++) {
    const x = (i / 8) * size + 2.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, size);
    ctx.stroke();
  }

  // Lower-edge dust grit speckles
  let seed = 314;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
  for (let i = 0; i < 480; i++) {
    const x = rand() * size;
    const y = size * (0.48 + rand() * 0.52);
    const r = 1.5 + rand() * 6.0;
    const a = (y / size - 0.45) * 0.95;
    ctx.fillStyle = `rgba(136, 56, 30, ${Math.min(0.48, a)})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.anisotropy = maxAnisotropy;
  return tex;
}

export function createContactShadowTexture(): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  const grad = ctx.createRadialGradient(size / 2, size / 2, 6, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(14, 6, 6, 0.82)');
  grad.addColorStop(0.52, 'rgba(18, 8, 7, 0.38)');
  grad.addColorStop(1, 'rgba(18, 8, 7, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Dark landing scorch / thruster blast & disturbed dust halo decal.
 */
export function createScorchDecalTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  const cx = size / 2;
  const cy = size / 2;

  const grad = ctx.createRadialGradient(cx, cy, 18, cx, cy, size * 0.49);
  grad.addColorStop(0.0, 'rgba(18, 10, 10, 0.92)');
  grad.addColorStop(0.55, 'rgba(32, 16, 14, 0.72)');
  grad.addColorStop(0.78, 'rgba(72, 32, 22, 0.42)');
  grad.addColorStop(1.0, 'rgba(154, 72, 42, 0.0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  // Radial blast streaks
  let seed = 919;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
  for (let i = 0; i < 54; i++) {
    const a = (i / 54) * Math.PI * 2 + (rand() - 0.5) * 0.12;
    const inner = 36 + rand() * 16;
    const outer = 88 + rand() * 38;
    ctx.strokeStyle = rand() > 0.3 ? 'rgba(20, 10, 9, 0.52)' : 'rgba(196, 102, 62, 0.28)';
    ctx.lineWidth = 2.5 + rand() * 4.0;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
    ctx.lineTo(cx + Math.cos(a) * outer, cy + Math.sin(a) * outer);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Tileable dual rover tire tracks + subtle astronaut bootprint trail decal.
 */
export function createRoverTrackTexture(maxAnisotropy: number): THREE.CanvasTexture {
  const width = 128;
  const height = 256;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, width, height);

  // Soft compacted regolith strips under left/right tracks
  const drawTrackStrip = (xCenter: number) => {
    ctx.fillStyle = 'rgba(42, 16, 10, 0.62)';
    ctx.fillRect(xCenter - 12, 0, 24, height);
    ctx.fillStyle = 'rgba(228, 142, 96, 0.28)';
    ctx.fillRect(xCenter - 16, 0, 4, height);
    ctx.fillRect(xCenter + 12, 0, 4, height);

    // Rover chevron cleat imprints
    ctx.strokeStyle = 'rgba(18, 7, 5, 0.72)';
    ctx.lineWidth = 2.5;
    for (let y = 8; y < height; y += 16) {
      ctx.beginPath();
      ctx.moveTo(xCenter - 10, y);
      ctx.lineTo(xCenter, y + 5);
      ctx.lineTo(xCenter + 10, y);
      ctx.stroke();
    }
  };

  drawTrackStrip(32);
  drawTrackStrip(96);

  // Subtle alternating astronaut bootprints between the rover tracks
  ctx.fillStyle = 'rgba(28, 11, 8, 0.58)';
  for (let i = 0; i < 6; i++) {
    const by = 18 + i * 40;
    const bx = i % 2 === 0 ? 57 : 71;
    ctx.beginPath();
    ctx.ellipse(bx, by, 3.6, 6.8, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = maxAnisotropy;
  return tex;
}

/**
 * Soft radial dust mote sprite for subtle airborne Martian dust particles.
 */
export function createDustParticleSpriteTexture(): THREE.CanvasTexture {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  const grad = ctx.createRadialGradient(size / 2, size / 2, 2, size / 2, size / 2, size / 2);
  grad.addColorStop(0.0, 'rgba(255, 224, 196, 0.92)');
  grad.addColorStop(0.45, 'rgba(235, 160, 112, 0.45)');
  grad.addColorStop(1.0, 'rgba(217, 123, 84, 0.0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function createDustOverlayTexture(maxAnisotropy: number): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, size, size);

  let seed = 808;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };

  // Mottled iron-oxide dust patches & fine grit
  for (let i = 0; i < 480; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const r = 2 + rand() * 11;
    const alpha = 0.08 + rand() * 0.24;
    ctx.fillStyle = rand() > 0.4 ? `rgba(184, 75, 43, ${alpha})` : `rgba(138, 52, 28, ${alpha})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Wind-swept red dust streaks
  ctx.strokeStyle = 'rgba(196, 88, 52, 0.22)';
  ctx.lineWidth = 2;
  for (let i = 0; i < 24; i++) {
    const y = rand() * size;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.quadraticCurveTo(size * 0.5, y + (rand() - 0.5) * 14, size, y);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(2, 2);
  tex.anisotropy = maxAnisotropy;
  return tex;
}
