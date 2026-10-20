// A photovoltaic module modelled on a typical datasheet of the 540-550 Wp
// class now installed at Indian utility parks such as Bhadla:
//   144 half-cut monocrystalline PERC cells (M10 wafers, 182 x 91 mm), 6 x 24,
//   wired as two halves of 72 with the bus ribbon across the middle,
//   10 busbars per cell, 3.2 mm AR-coated tempered glass,
//   2279 x 1134 x 35 mm, anodised aluminium frame with a C profile,
//   3 split junction boxes on the back, 4 mm2 cables with MC4 connectors.
// Every surface is drawn procedurally into texture maps at real scale.
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

// ---- datasheet geometry (millimetres) -------------------------------------------
const COLS = 6, HALF_ROWS = 12;
const CELL_W = 182, CELL_H = 91;
const GAP_COL = 3, GAP_ROW = 2, CENTER_GAP = 12;
export const PANEL_W = 1134, PANEL_H = 2279, PANEL_T = 35;
const ARRAY_W = COLS * CELL_W + (COLS - 1) * GAP_COL;
const HALF_H = HALF_ROWS * CELL_H + (HALF_ROWS - 1) * GAP_ROW;
const MARGIN_X = (PANEL_W - ARRAY_W) / 2;
const MARGIN_Y = (PANEL_H - 2 * HALF_H - CENTER_GAP) / 2;
const BUSBARS = 10;
const MM = 1 / 1000;

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function cells() {
  const out = [];
  for (let half = 0; half < 2; half++) {
    const y0 = MARGIN_Y + half * (HALF_H + CENTER_GAP);
    for (let r = 0; r < HALF_ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        // Half-cut cells keep the wafer's rounded corners on one long side only.
        const roundTop = (r + half) % 2 === 0;
        out.push({ x: MARGIN_X + c * (CELL_W + GAP_COL), y: y0 + r * (CELL_H + GAP_ROW), roundTop });
      }
    }
  }
  return out;
}

function cellPath(ctx, x, y, roundTop) {
  const k = 4.5; // M10 corner radius, seen as a small cut
  ctx.beginPath();
  if (roundTop) {
    ctx.moveTo(x + k, y); ctx.lineTo(x + CELL_W - k, y); ctx.lineTo(x + CELL_W, y + k);
    ctx.lineTo(x + CELL_W, y + CELL_H); ctx.lineTo(x, y + CELL_H); ctx.lineTo(x, y + k);
  } else {
    ctx.moveTo(x, y); ctx.lineTo(x + CELL_W, y); ctx.lineTo(x + CELL_W, y + CELL_H - k);
    ctx.lineTo(x + CELL_W - k, y + CELL_H); ctx.lineTo(x + k, y + CELL_H); ctx.lineTo(x, y + CELL_H - k);
  }
  ctx.closePath();
}

function canvas(pxPerMM, w = PANEL_W, h = PANEL_H) {
  const cv = document.createElement("canvas");
  cv.width = Math.round(w * pxPerMM);
  cv.height = Math.round(h * pxPerMM);
  const ctx = cv.getContext("2d");
  ctx.scale(pxPerMM, pxPerMM);
  return [cv, ctx];
}

// Draws one map of the front. `paint` decides the value for each material.
function drawFront(pxPerMM, paint, seed = 11) {
  const [cv, ctx] = canvas(pxPerMM);
  const rand = rng(seed);
  ctx.fillStyle = paint.backsheet;
  ctx.fillRect(0, 0, PANEL_W, PANEL_H);

  const busXs = Array.from({ length: BUSBARS }, (_, i) => (i + 0.5) / BUSBARS);
  for (const cell of cells()) {
    const { x, y, roundTop } = cell;
    cellPath(ctx, x, y, roundTop);
    ctx.fillStyle = paint.cell(rand);
    ctx.fill();
    if (paint.grain) {
      // Alkaline texturing leaves a fine, slightly sparkly pyramid surface.
      ctx.save();
      cellPath(ctx, x, y, roundTop);
      ctx.clip();
      for (let i = 0; i < 260; i++) {
        ctx.fillStyle = paint.grain(rand);
        ctx.fillRect(x + rand() * CELL_W, y + rand() * CELL_H, 0.5, 0.5);
      }
      ctx.restore();
    }
    // Fingers: fine silver lines across the cell, perpendicular to the busbars.
    ctx.fillStyle = paint.finger;
    for (let fy = y + 0.9; fy < y + CELL_H - 0.5; fy += 1.25) ctx.fillRect(x + 0.6, fy, CELL_W - 1.2, 0.09);
  }
  // Busbar wires run the length of each string, through both halves' cells.
  ctx.fillStyle = paint.busbar;
  for (let c = 0; c < COLS; c++) {
    const x0 = MARGIN_X + c * (CELL_W + GAP_COL);
    for (const b of busXs) {
      for (let half = 0; half < 2; half++) {
        const y0 = MARGIN_Y + half * (HALF_H + CENTER_GAP);
        ctx.fillRect(x0 + b * CELL_W - 0.22, y0 - 4, 0.44, HALF_H + 8);
      }
    }
  }
  // Bus ribbons: top, bottom and the shared centre bus of a half-cut module.
  ctx.fillStyle = paint.ribbon;
  const ribbon = (y) => ctx.fillRect(MARGIN_X - 1, y, ARRAY_W + 2, 4.5);
  ribbon(MARGIN_Y - 9);
  ribbon(PANEL_H - MARGIN_Y + 4.5);
  ribbon(MARGIN_Y + HALF_H + CENTER_GAP / 2 - 2.2);
  return cv;
}

function heightToNormal(src, strength) {
  const w = src.width, h = src.height;
  const d = src.getContext("2d").getImageData(0, 0, w, h).data;
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  const out = ctx.createImageData(w, h);
  const H = (x, y) => d[(Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))) * 4] / 255;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = (H(x + 1, y) - H(x - 1, y)) * strength;
      const dy = (H(x, y + 1) - H(x, y - 1)) * strength;
      const len = Math.hypot(dx, dy, 1);
      const i = (y * w + x) * 4;
      out.data[i] = (-dx / len * 0.5 + 0.5) * 255;
      out.data[i + 1] = (dy / len * 0.5 + 0.5) * 255;
      out.data[i + 2] = (1 / len * 0.5 + 0.5) * 255;
      out.data[i + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
  return cv;
}

// Solar glass has a faint "prismatic" dimple on its surface.
function glassNormal(size = 512) {
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

// Dust coverage (alphaMap reads green: black = clean, white = covered).
// The panel stands in portrait, so dust gathers toward the bottom edge,
// with a dried dirt line against the lower frame and rain channels.
function dust(pxPerMM, seed = 3) {
  const [cv, ctx] = canvas(pxPerMM);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, PANEL_W, PANEL_H);
  const rand = rng(seed);
  const down = (y) => y / PANEL_H;
  for (let i = 0; i < 520; i++) {
    const y = PANEL_H * Math.pow(rand(), 0.7), x = rand() * PANEL_W, r = 40 + rand() * 170;
    const a = (0.02 + rand() * 0.045) * (0.45 + down(y));
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(255,255,255,${a})`);
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
  }
  for (let i = 0; i < 42000; i++) {
    const y = PANEL_H * Math.pow(rand(), 0.55), x = rand() * PANEL_W;
    const r = 0.2 + Math.pow(rand(), 4) * 1.5;
    ctx.fillStyle = `rgba(255,255,255,${0.2 + rand() * 0.5})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  for (let i = 0; i < 120; i++) {
    const x = rand() * PANEL_W, len = 80 + rand() * 420, y = PANEL_H - 30 - len * rand();
    const g = ctx.createLinearGradient(0, y, 0, y + len);
    g.addColorStop(0, "rgba(255,255,255,0)");
    g.addColorStop(1, `rgba(255,255,255,${0.05 + rand() * 0.1})`);
    ctx.fillStyle = g;
    ctx.fillRect(x, y, 1.5 + rand() * 5, len);
  }
  const band = ctx.createLinearGradient(0, PANEL_H - 60, 0, PANEL_H);
  band.addColorStop(0, "rgba(255,255,255,0)");
  band.addColorStop(0.55, "rgba(255,255,255,0.35)");
  band.addColorStop(1, "rgba(255,255,255,0.8)");
  ctx.fillStyle = band;
  ctx.fillRect(0, PANEL_H - 60, PANEL_W, 60);
  return cv;
}

// Backsheet: white polymer with a faint woven texture and the rating label.
function backsheet(pxPerMM) {
  const [cv, ctx] = canvas(pxPerMM);
  ctx.fillStyle = "#d6d7d3";
  ctx.fillRect(0, 0, PANEL_W, PANEL_H);
  const rand = rng(5);
  for (let i = 0; i < 26000; i++) {
    ctx.fillStyle = `rgba(0,0,0,${rand() * 0.035})`;
    ctx.fillRect(rand() * PANEL_W, rand() * PANEL_H, 1 + rand() * 3, 0.5);
  }
  // Faint shadows of the cells through the backsheet.
  ctx.fillStyle = "rgba(0,0,0,0.035)";
  for (const c of cells()) ctx.fillRect(PANEL_W - c.x - CELL_W, c.y, CELL_W, CELL_H);
  // Rating label (values typical of this module class).
  const lx = PANEL_W / 2 - 110, ly = PANEL_H * 0.72, lw = 220, lh = 150;
  ctx.fillStyle = "#f7f7f4";
  ctx.fillRect(lx, ly, lw, lh);
  ctx.fillStyle = "#1b1b1b";
  ctx.font = "600 13px Manrope, sans-serif";
  ctx.fillText("Monocrystalline PERC module", lx + 12, ly + 22);
  ctx.font = "400 9px Manrope, sans-serif";
  const lines = [
    "144 half-cut cells (M10, 182 mm)",
    "Rated power Pmax          545 W",
    "Voltage at Pmax Vmp      41.8 V",
    "Current at Pmax Imp     13.04 A",
    "Open-circuit voltage Voc 49.6 V",
    "Short-circuit current Isc 13.9 A",
    "Max system voltage    1500 V DC",
    "Dimensions  2279 x 1134 x 35 mm",
    "STC: 1000 W/m2, AM 1.5, 25 C",
  ];
  lines.forEach((t, i) => ctx.fillText(t, lx + 12, ly + 42 + i * 11.5));
  ctx.fillStyle = "#c9c9c4";
  ctx.fillRect(lx + 12, ly + lh - 14, lw - 24, 6);
  return cv;
}

function tex(cv, renderer, srgb = false) {
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}

// C-shaped frame profile (metres), extruded along each edge.
function frameProfile() {
  const s = new THREE.Shape();
  const T = PANEL_T * MM, lip = 0.011, flange = 0.024, wall = 0.0018;
  // x: outward (0) to inward, y: back (0) to front (T)
  s.moveTo(0, 0);
  s.lineTo(flange, 0);
  s.lineTo(flange, wall);
  s.lineTo(wall, wall);
  s.lineTo(wall, T - 0.0062);
  s.lineTo(lip, T - 0.0062);
  s.lineTo(lip, T - 0.0042);
  s.lineTo(wall + 0.001, T - 0.0042);
  s.lineTo(wall + 0.001, T - 0.0006);
  s.lineTo(lip + 0.0012, T - 0.0006);
  s.lineTo(lip + 0.0012, T);
  s.lineTo(0, T);
  s.closePath();
  return s;
}

function frameBar(length, material) {
  const geo = new THREE.ExtrudeGeometry(frameProfile(), {
    depth: length, bevelEnabled: true, bevelThickness: 0.0004, bevelSize: 0.0004, bevelSegments: 2, steps: 1,
  });
  geo.translate(0, -PANEL_T * MM / 2, -length / 2);
  const m = new THREE.Mesh(geo, material);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

function cable(material, from, to, sag = 0.18) {
  const mid = from.clone().lerp(to, 0.5);
  mid.z -= sag * 0.35;
  mid.y -= sag;
  const curve = new THREE.CatmullRomCurve3([from, from.clone().add(new THREE.Vector3(0, -0.04, -0.02)), mid, to]);
  const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, 0.0035, 10, false), material);
  m.castShadow = true;
  return m;
}

export function createPanelKit(renderer, { quality = "high" } = {}) {
  const px = quality === "high" ? 1.5 : 0.9;
  const t0 = performance.now();

  const color = drawFront(px, {
    backsheet: "#dfe2e4",
    cell: (r) => { const v = 0.88 + r() * 0.12; return `rgb(${Math.round(11 * v)},${Math.round(17 * v)},${Math.round(33 * v)})`; },
    grain: (r) => `rgba(${r() < 0.5 ? "80,100,150" : "0,0,0"},${0.06 + r() * 0.1})`,
    finger: "rgba(140,148,160,0.5)",
    busbar: "#cfd3d8",
    ribbon: "#c8ccd1",
  });
  const rough = drawFront(px, {
    backsheet: "#cfcfcf", cell: () => "#808080", finger: "#555", busbar: "#383838", ribbon: "#3a3a3a",
  });
  const metal = drawFront(px, {
    backsheet: "#000", cell: () => "#000", finger: "#888", busbar: "#fff", ribbon: "#fff",
  });
  const height = drawFront(px * 0.7, {
    backsheet: "#1a1a1a", cell: () => "#000", finger: "#4a4a4a", busbar: "#ffffff", ribbon: "#d0d0d0",
  });

  const glassN = tex(glassNormal(), renderer);
  glassN.wrapS = glassN.wrapT = THREE.RepeatWrapping;
  glassN.repeat.set(5, 10);

  const cellMat = new THREE.MeshPhysicalMaterial({
    map: tex(color, renderer, true),
    roughnessMap: tex(rough, renderer),
    metalnessMap: tex(metal, renderer),
    normalMap: tex(heightToNormal(height, 3.2), renderer),
    normalScale: new THREE.Vector2(0.55, 0.55),
    roughness: 1, metalness: 1,
    clearcoat: 1,                       // 3.2 mm tempered glass
    clearcoatRoughness: 0.03,
    clearcoatNormalMap: glassN,
    clearcoatNormalScale: new THREE.Vector2(0.07, 0.07),
    specularIntensity: 0.55,
    iridescence: 0.18, iridescenceIOR: 1.33, iridescenceThicknessRange: [150, 400], // AR coating sheen
  });
  const dustMat = new THREE.MeshStandardMaterial({
    color: "#c9b796", alphaMap: tex(dust(px * 0.8), renderer),
    transparent: true, opacity: 0, roughness: 1, metalness: 0, depthWrite: false,
  });
  const frameMat = new THREE.MeshPhysicalMaterial({
    color: "#bfc3c8", metalness: 1, roughness: 0.3, anisotropy: 0.6, clearcoat: 0.25, clearcoatRoughness: 0.2,
  });
  const backMat = new THREE.MeshStandardMaterial({ map: tex(backsheet(1.0), renderer, true), roughness: 0.82 });
  const boxMat = new THREE.MeshPhysicalMaterial({ color: "#111214", roughness: 0.45, clearcoat: 0.4, clearcoatRoughness: 0.4 });
  const cableMat = new THREE.MeshStandardMaterial({ color: "#0c0c0d", roughness: 0.38 });
  const mc4Mat = new THREE.MeshStandardMaterial({ color: "#151517", roughness: 0.3, metalness: 0.1 });

  const W = PANEL_W * MM, H = PANEL_H * MM, T = PANEL_T * MM;
  const panel = new THREE.Group();

  const face = new THREE.Mesh(new THREE.PlaneGeometry(W - 0.002, H - 0.002), cellMat);
  face.position.z = T / 2 - 0.0042;
  face.receiveShadow = true;
  const dustMesh = new THREE.Mesh(face.geometry, dustMat);
  dustMesh.position.z = face.position.z + 0.0005;
  panel.add(face, dustMesh);

  // Frame: four extruded C-profiles. The profile is drawn with x = inward
  // from the outer edge and y = depth (back to front), extruded along z; each
  // side maps those axes onto the panel with an explicit right-handed basis.
  const Z = new THREE.Vector3(0, 0, 1);
  const sides = [
    { len: H, inward: new THREE.Vector3(1, 0, 0), at: new THREE.Vector3(-W / 2, 0, 0) },
    { len: H, inward: new THREE.Vector3(-1, 0, 0), at: new THREE.Vector3(W / 2, 0, 0) },
    { len: W, inward: new THREE.Vector3(0, -1, 0), at: new THREE.Vector3(0, H / 2, 0) },
    { len: W, inward: new THREE.Vector3(0, 1, 0), at: new THREE.Vector3(0, -H / 2, 0) },
  ];
  for (const s of sides) {
    const bar = frameBar(s.len, frameMat);
    const along = new THREE.Vector3().crossVectors(s.inward, Z);
    bar.geometry.applyMatrix4(new THREE.Matrix4().makeBasis(s.inward, Z, along));
    bar.position.copy(s.at);
    panel.add(bar);
  }

  const back = new THREE.Mesh(new THREE.PlaneGeometry(W - 0.004, H - 0.004), backMat);
  back.rotation.y = Math.PI;
  back.position.z = -T / 2 + 0.006;
  back.receiveShadow = true;
  panel.add(back);

  // Three split junction boxes across the middle, two cables with MC4 ends.
  const boxGeo = new RoundedBoxGeometry(0.075, 0.055, 0.018, 3, 0.004);
  const boxes = [-0.3, 0, 0.3].map((x) => {
    const b = new THREE.Mesh(boxGeo, boxMat);
    b.position.set(x, 0, -T / 2 - 0.004);
    b.castShadow = true;
    panel.add(b);
    return b;
  });
  for (const [b, dir] of [[boxes[0], -1], [boxes[2], 1]]) {
    const start = b.position.clone().add(new THREE.Vector3(dir * 0.03, -0.02, -0.006));
    const end = new THREE.Vector3(b.position.x + dir * 0.16, -0.52, -T / 2 - 0.05);
    panel.add(cable(cableMat, start, end, 0.12));
    const plug = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.06, 20), mc4Mat);
    const ring = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, 0.012, 20), mc4Mat);
    ring.position.y = 0.022;
    plug.add(body, ring);
    plug.position.copy(end).add(new THREE.Vector3(0, -0.03, 0));
    plug.castShadow = true;
    panel.add(plug);
  }

  panel.traverse((o) => { if (o.isMesh && o.material !== dustMat) o.castShadow = true; });
  console.debug(`panel textures built in ${Math.round(performance.now() - t0)} ms`);

  return {
    panel,
    width: W, height: H, thickness: T,
    setSoiling(level) { dustMat.opacity = level; dustMat.visible = level > 0.001; },
  };
}
