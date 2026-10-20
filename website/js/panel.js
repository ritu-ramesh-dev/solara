// Procedural photovoltaic module: a 60-cell monocrystalline panel built from
// canvas-drawn texture maps (colour, roughness, metalness, normal) plus a
// separate dust layer whose opacity follows the soiling level.
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

// Real-world dimensions in millimetres (typical 60-cell module).
const COLS = 6, ROWS = 10;
const CELL = 156, GAP = 3, MARGIN_X = 14, MARGIN_Y = 40;
export const PANEL_W = COLS * CELL + (COLS - 1) * GAP + 2 * MARGIN_X;   // ~ 1010 mm
export const PANEL_H = ROWS * CELL + (ROWS - 1) * GAP + 2 * MARGIN_Y;   // ~ 1667 mm
const MM = 1 / 1000; // scene units are metres

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function cellPath(ctx, x, y, s, chamfer) {
  ctx.beginPath();
  ctx.moveTo(x + chamfer, y);
  ctx.lineTo(x + s - chamfer, y);
  ctx.lineTo(x + s, y + chamfer);
  ctx.lineTo(x + s, y + s - chamfer);
  ctx.lineTo(x + s - chamfer, y + s);
  ctx.lineTo(x + chamfer, y + s);
  ctx.lineTo(x, y + s - chamfer);
  ctx.lineTo(x, y + chamfer);
  ctx.closePath();
}

function cellOrigins() {
  const out = [];
  for (let r = 0; r < ROWS; r++)
    for (let c = 0; c < COLS; c++)
      out.push([MARGIN_X + c * (CELL + GAP), MARGIN_Y + r * (CELL + GAP), r, c]);
  return out;
}

// Draws one map. `paint` decides the value of each kind of surface.
function drawMap(pxPerMM, paint, seed = 7) {
  const w = Math.round(PANEL_W * pxPerMM), h = Math.round(PANEL_H * pxPerMM);
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  ctx.scale(pxPerMM, pxPerMM);
  const rand = rng(seed);
  const chamfer = 11;
  const busX = [0.1, 0.3, 0.5, 0.7, 0.9]; // 5 busbars per cell

  // Backsheet (seen through the gaps between cells).
  ctx.fillStyle = paint.backsheet;
  ctx.fillRect(0, 0, PANEL_W, PANEL_H);

  for (const [x, y] of cellOrigins()) {
    // Cell body with slight per-cell tone variation (wafers are never identical).
    cellPath(ctx, x, y, CELL, chamfer);
    ctx.fillStyle = paint.cell(rand);
    ctx.fill();

    // Fine grain: the textured silicon surface.
    if (paint.grain) {
      ctx.save();
      cellPath(ctx, x, y, CELL, chamfer);
      ctx.clip();
      for (let i = 0; i < 900; i++) {
        ctx.fillStyle = paint.grain(rand);
        ctx.fillRect(x + rand() * CELL, y + rand() * CELL, 0.6, 0.6);
      }
      ctx.restore();
    }

    // Fingers: thin horizontal silver lines across the cell.
    ctx.fillStyle = paint.finger;
    for (let fy = y + 1.2; fy < y + CELL - 1; fy += 1.7) {
      const inset = (fy - y < chamfer) ? chamfer - (fy - y)
        : (y + CELL - fy < chamfer) ? chamfer - (y + CELL - fy) : 0;
      ctx.fillRect(x + inset + 0.6, fy, CELL - 2 * inset - 1.2, 0.12);
    }
  }

  // Busbars and the ribbons that link cells within each vertical string.
  ctx.fillStyle = paint.busbar;
  for (let c = 0; c < COLS; c++) {
    const x0 = MARGIN_X + c * (CELL + GAP);
    for (const b of busX) ctx.fillRect(x0 + b * CELL - 0.55, MARGIN_Y - 9, 1.1, PANEL_H - 2 * MARGIN_Y + 18);
  }
  // Cross-ribbons at top and bottom joining the strings in series.
  for (let c = 0; c < COLS; c += 2) {
    const x0 = MARGIN_X + c * (CELL + GAP) + busX[0] * CELL - 2;
    const x1 = MARGIN_X + (c + 1) * (CELL + GAP) + busX[4] * CELL + 2;
    ctx.fillRect(x0, PANEL_H - MARGIN_Y + 7, x1 - x0, 5);
  }
  for (let c = 1; c < COLS - 1; c += 2) {
    const x0 = MARGIN_X + c * (CELL + GAP) + busX[0] * CELL - 2;
    const x1 = MARGIN_X + (c + 1) * (CELL + GAP) + busX[4] * CELL + 2;
    ctx.fillRect(x0, MARGIN_Y - 12, x1 - x0, 5);
  }
  const first = MARGIN_X + busX[0] * CELL - 2, last = PANEL_W - MARGIN_X - (1 - busX[4]) * CELL + 2;
  ctx.fillRect(first, MARGIN_Y - 12, MARGIN_X + busX[4] * CELL + 2 - first, 5);
  ctx.fillRect(PANEL_W - MARGIN_X - CELL + busX[0] * CELL - 2, MARGIN_Y - 12, last - (PANEL_W - MARGIN_X - CELL + busX[0] * CELL - 2), 5);

  return cv;
}

// Height map -> tangent-space normal map (for the raised busbars and fingers).
function heightToNormal(heightCanvas, strength = 2.2) {
  const w = heightCanvas.width, h = heightCanvas.height;
  const src = heightCanvas.getContext("2d").getImageData(0, 0, w, h).data;
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  const out = ctx.createImageData(w, h);
  const H = (x, y) => src[((Math.min(h - 1, Math.max(0, y)) * w) + Math.min(w - 1, Math.max(0, x))) * 4] / 255;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = (H(x + 1, y) - H(x - 1, y)) * strength;
      const dy = (H(x, y + 1) - H(x, y - 1)) * strength;
      const len = Math.hypot(dx, dy, 1);
      const i = (y * w + x) * 4;
      out.data[i] = ((-dx / len) * 0.5 + 0.5) * 255;
      out.data[i + 1] = ((dy / len) * 0.5 + 0.5) * 255;
      out.data[i + 2] = ((1 / len) * 0.5 + 0.5) * 255;
      out.data[i + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
  return cv;
}

// Faint dimpled pattern of solar glass, used as the clearcoat normal map.
function glassDimpleNormal(size = 512) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  const rand = rng(99);
  for (let i = 0; i < 2600; i++) {
    const x = rand() * size, y = rand() * size, r = 2 + rand() * 5;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, "rgba(255,255,255,0.35)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
  }
  return heightToNormal(cv, 1.2);
}

// Dust coverage map (alphaMap reads the green channel: black = clean glass,
// white = fully covered). The panel is mounted in landscape, so the canvas's
// left edge (x = 0) is the panel's lower edge, where dust and dirt collect.
function dustCanvas(pxPerMM, seed = 3) {
  const w = Math.round(PANEL_W * pxPerMM), h = Math.round(PANEL_H * pxPerMM);
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  ctx.scale(pxPerMM, pxPerMM);
  const rand = rng(seed);
  const downness = (x) => 1 - x / PANEL_W; // 1 at the lower edge, 0 at the top

  // Uneven haze of fine dust, thicker toward the lower edge.
  for (let i = 0; i < 420; i++) {
    const x = PANEL_W * Math.pow(rand(), 1.4), y = rand() * PANEL_H, r = 30 + rand() * 140;
    const a = (0.025 + rand() * 0.05) * (0.5 + downness(x));
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(255,255,255,${a})`);
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
  }
  // Grains of sand and dust.
  for (let i = 0; i < 30000; i++) {
    const x = PANEL_W * Math.pow(rand(), 1.8);
    const y = rand() * PANEL_H;
    const r = 0.2 + Math.pow(rand(), 4) * 1.4;
    ctx.fillStyle = `rgba(255,255,255,${0.2 + rand() * 0.5})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  // Rain streaks: cleaner channels with dirt pushed to their edges, running
  // down the slope toward the lower frame.
  for (let i = 0; i < 90; i++) {
    const y = rand() * PANEL_H, len = 60 + rand() * 340, x = rand() * 160;
    const g = ctx.createLinearGradient(x + len, 0, x, 0);
    g.addColorStop(0, "rgba(255,255,255,0)");
    g.addColorStop(1, `rgba(255,255,255,${0.06 + rand() * 0.1})`);
    ctx.fillStyle = g;
    ctx.fillRect(x, y, len, 1.5 + rand() * 5);
  }
  // Dirt line where water evaporates against the lower frame.
  const band = ctx.createLinearGradient(0, 0, 55, 0);
  band.addColorStop(0, "rgba(255,255,255,0.75)");
  band.addColorStop(0.45, "rgba(255,255,255,0.35)");
  band.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = band;
  ctx.fillRect(0, 0, 55, PANEL_H);
  return cv;
}

function tex(canvas, renderer, srgb = false) {
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  return t;
}

export function createPanelKit(renderer, { quality = "high" } = {}) {
  const px = quality === "high" ? 1.6 : 1.0;

  const color = drawMap(px, {
    backsheet: "#d9dcdf",
    cell: (r) => { const v = 0.9 + r() * 0.12; return `rgb(${Math.round(14 * v)},${Math.round(22 * v)},${Math.round(44 * v)})`; },
    grain: (r) => `rgba(${r() < 0.5 ? "70,90,140" : "0,0,0"},${0.05 + r() * 0.08})`,
    finger: "rgba(150,160,175,0.55)",
    busbar: "#c4c9cf",
  });
  const rough = drawMap(px, {
    backsheet: "#c8c8c8", cell: () => "#7a7a7a", finger: "#4a4a4a", busbar: "#3a3a3a",
  });
  const metal = drawMap(px, {
    backsheet: "#000", cell: () => "#000", finger: "#9a9a9a", busbar: "#ffffff",
  });
  const height = drawMap(px * 0.75, {
    backsheet: "#202020", cell: () => "#000", finger: "#5a5a5a", busbar: "#ffffff",
  });

  const cellMat = new THREE.MeshPhysicalMaterial({
    map: tex(color, renderer, true),
    roughnessMap: tex(rough, renderer),
    metalnessMap: tex(metal, renderer),
    normalMap: tex(heightToNormal(height, 3.0), renderer),
    normalScale: new THREE.Vector2(0.6, 0.6),
    roughness: 1, metalness: 1,
    // The tempered glass sits on top as a clearcoat with a faint dimple pattern.
    clearcoat: 1,
    clearcoatRoughness: 0.035,
    clearcoatNormalMap: (() => { const t = tex(glassDimpleNormal(), renderer); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(6, 10); return t; })(),
    clearcoatNormalScale: new THREE.Vector2(0.08, 0.08),
    specularIntensity: 0.6,
    iridescence: 0.12, iridescenceIOR: 1.35, iridescenceThicknessRange: [180, 420], // anti-reflective coating tint
  });

  const dustMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color("#c2b294"),
    alphaMap: tex(dustCanvas(px * 0.8), renderer),
    transparent: true,
    opacity: 0,
    roughness: 1,
    metalness: 0,
    depthWrite: false,
  });

  const frameMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color("#9fa4aa"),
    metalness: 1,
    roughness: 0.45,
    anisotropy: 0.4,
  });
  const backMat = new THREE.MeshStandardMaterial({ color: "#e7e7e5", roughness: 0.85 });
  const boxMat = new THREE.MeshStandardMaterial({ color: "#151515", roughness: 0.6 });

  const W = PANEL_W * MM, H = PANEL_H * MM, T = 0.035, LIP = 0.032;
  const faceGeo = new THREE.PlaneGeometry(W - 0.004, H - 0.004);
  const geos = {
    face: faceGeo,
    long: new RoundedBoxGeometry(LIP, H + LIP, T, 3, 0.003),
    short: new RoundedBoxGeometry(W + LIP, LIP, T, 3, 0.003),
    back: new THREE.PlaneGeometry(W, H),
    jbox: new RoundedBoxGeometry(0.16, 0.12, 0.025, 2, 0.006),
  };

  function makePanel() {
    const g = new THREE.Group();
    const face = new THREE.Mesh(geos.face, cellMat);
    face.position.z = T / 2 - 0.004;
    face.receiveShadow = true;
    const dust = new THREE.Mesh(geos.face, dustMat);
    dust.position.z = face.position.z + 0.0006;
    dust.receiveShadow = true;
    g.add(face, dust);

    for (const s of [-1, 1]) {
      const l = new THREE.Mesh(geos.long, frameMat);
      l.position.set(s * (W / 2), 0, 0);
      const sh = new THREE.Mesh(geos.short, frameMat);
      sh.position.set(0, s * (H / 2), 0);
      for (const m of [l, sh]) { m.castShadow = true; m.receiveShadow = true; g.add(m); }
    }
    const back = new THREE.Mesh(geos.back, backMat);
    back.rotation.y = Math.PI;
    back.position.z = -T / 2 + 0.002;
    const jbox = new THREE.Mesh(geos.jbox, boxMat);
    jbox.position.set(0, H * 0.36, -T / 2 - 0.012);
    g.add(back, jbox);
    return g;
  }

  return {
    makePanel,
    width: W, height: H, thickness: T,
    setSoiling(level) { dustMat.opacity = level; dustMat.visible = level > 0.001; },
    materials: { cellMat, dustMat, frameMat },
  };
}
