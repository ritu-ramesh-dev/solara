// The hero scene: a row of solar panels in darkness, lit only by a small sun
// that follows the cursor (or drifts on its own on touch screens).
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { createPanelKit } from "./panel.js";

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const finePointer = matchMedia("(pointer: fine)").matches;

export function startScene(canvas, { onReady } = {}) {
  const small = Math.min(innerWidth, innerHeight) < 700;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, small ? 1.5 : 2));
  renderer.setSize(innerWidth, innerHeight, false);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#050505");
  scene.fog = new THREE.FogExp2("#050505", 0.12);

  // A very dim studio environment so the glass has something to reflect.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.035;

  const camera = new THREE.PerspectiveCamera(30, innerWidth / innerHeight, 0.05, 60);
  const lookAt = new THREE.Vector3(0, 0, 0);

  // ---- panels -------------------------------------------------------------
  const kit = createPanelKit(renderer, { quality: small ? "low" : "high" });
  const rig = new THREE.Group();
  scene.add(rig);

  const TILT = -0.62; // panels lean back toward the sky
  const pitch = kit.height + 0.02; // landscape: panel's long side runs along the row
  const rows = [{ z: 0, y: 0 }, { z: -2.4, y: 0.05 }];
  let hero = null;
  const steel = new THREE.MeshStandardMaterial({ color: "#5d6166", metalness: 0.9, roughness: 0.5 });
  for (const row of rows) {
    const rowGroup = new THREE.Group();
    rowGroup.position.set(0, row.y, row.z);
    for (let i = -3; i <= 3; i++) {
      const p = kit.makePanel();
      p.rotation.z = Math.PI / 2;           // landscape
      const holder = new THREE.Group();
      holder.add(p);
      holder.rotation.x = TILT;
      holder.position.x = i * pitch;
      rowGroup.add(holder);
      if (row.z === 0 && i === 0) hero = holder;
    }
    // Two rails behind the panels and posts down to the ground.
    const railGeo = new THREE.CylinderGeometry(0.02, 0.02, pitch * 7 + 0.4, 12);
    for (const off of [-0.28, 0.28]) {
      const rail = new THREE.Mesh(railGeo, steel);
      rail.rotation.z = Math.PI / 2;
      const local = new THREE.Vector3(0, off, -0.06).applyEuler(new THREE.Euler(TILT, 0, 0));
      rail.position.copy(local);
      rail.castShadow = true;
      rowGroup.add(rail);
    }
    const postGeo = new THREE.CylinderGeometry(0.03, 0.03, 1.6, 12);
    for (let i = -3; i <= 3; i += 2) {
      for (const [off, len] of [[-0.28, 1.25], [0.28, 0.95]]) {
        const local = new THREE.Vector3(0, off, -0.06).applyEuler(new THREE.Euler(TILT, 0, 0));
        const post = new THREE.Mesh(postGeo, steel);
        post.scale.y = len / 1.6;
        post.position.set(i * pitch + 0.25, local.y - len / 2, local.z);
        post.castShadow = true;
        rowGroup.add(post);
      }
    }
    rig.add(rowGroup);
  }

  rig.traverse((o) => { if (o.isMesh && o.material !== kit.materials.dustMat) o.castShadow = true; });

  // Gravel ground catches the pool of light.
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.MeshStandardMaterial({ color: "#1a1816", roughness: 0.95 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -1.05;
  ground.receiveShadow = true;
  rig.add(ground);

  // ---- the sun ---------------------------------------------------------------
  const SUN = 0.9;
  const sun = new THREE.PointLight("#fff0dc", SUN, 2.6, 2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(small ? 512 : 1024, small ? 512 : 1024);
  sun.shadow.bias = -0.0008;
  sun.shadow.normalBias = 0.01;
  sun.shadow.radius = 4;
  const orb = new THREE.Mesh(
    new THREE.SphereGeometry(0.016, 32, 16),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(7, 6.4, 5.6), toneMapped: false }),
  );
  sun.add(orb);
  scene.add(sun);
  scene.add(new THREE.HemisphereLight("#2a3140", "#0a0806", 0.025));

  // ---- post-processing -------------------------------------------------------
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth / 2, innerHeight / 2), 0.55, 0.5, 1.8);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  // ---- pointer -> light ------------------------------------------------------
  const ndc = new THREE.Vector2(0, 0.15);
  let lastMove = -1e9;
  const raycaster = new THREE.Raycaster();
  const lightPlane = new THREE.Plane();
  const target = new THREE.Vector3();
  const current = new THREE.Vector3(0.2, 0.35, 0.9);
  const tmpN = new THREE.Vector3(), tmpP = new THREE.Vector3();

  addEventListener("pointermove", (e) => {
    if (e.pointerType !== "mouse" && e.pointerType !== "pen") return;
    ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    lastMove = performance.now();
  }, { passive: true });

  // ---- scroll choreography ----------------------------------------------------
  let scrollP = 0;
  const onScroll = () => { scrollP = scrollY / innerHeight; };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const soil = { value: 0, target: 0 };

  let camDist = 5;
  function layoutCamera() {
    const aspect = innerWidth / innerHeight;
    camera.aspect = aspect;
    // Pull back on narrow screens so the whole panel stays in frame.
    camDist = aspect < 0.8 ? 8.2 : aspect < 1.2 ? 7 : aspect < 1.5 ? 6.2 : 6.4;
    camera.updateProjectionMatrix();
  }
  layoutCamera();

  addEventListener("resize", () => {
    renderer.setSize(innerWidth, innerHeight, false);
    composer.setSize(innerWidth, innerHeight);
    bloom.setSize(innerWidth / 2, innerHeight / 2);
    layoutCamera();
  });

  const clock = new THREE.Clock();
  const ease = (t) => t * t * (3 - 2 * t);
  let ready = false;

  function frame() {
    const dt = Math.min(clock.getDelta(), 0.05);
    const t = clock.elapsedTime;

    // Scene placement: centred on the first screen, slides right for the second.
    const s = ease(THREE.MathUtils.clamp(scrollP, 0, 1));
    const wide = innerWidth / innerHeight > 1.1;
    rig.position.x = wide ? THREE.MathUtils.lerp(0, 2.0, s) : 0;
    rig.rotation.y = THREE.MathUtils.lerp(-0.2, -0.38, s);
    camera.position.set(0, THREE.MathUtils.lerp(1.25, 0.8, s), camDist * THREE.MathUtils.lerp(1, 0.8, s));
    camera.lookAt(lookAt);
    // Shift the picture up so the panel sits above the title (like the statue).
    const lift = (wide ? 0.2 : 0.13) * (1 - s) + (wide ? 0 : 0.2) * s;
    camera.setViewOffset(innerWidth, innerHeight, 0, innerHeight * lift, innerWidth, innerHeight);
    canvas.style.opacity = String(THREE.MathUtils.clamp(1.9 - scrollP, 0.22, 1));

    // Light plane: parallel to the hero panel, 0.55 m in front of it.
    hero.updateWorldMatrix(true, false);
    tmpN.set(0, 0, 1).transformDirection(hero.matrixWorld);
    hero.getWorldPosition(tmpP);
    lightPlane.setFromNormalAndCoplanarPoint(tmpN, tmpP.addScaledVector(tmpN, 0.38));

    const idle = !finePointer || performance.now() - lastMove > 5000;
    if (idle) {
      // The sun drifts in a slow arc across the hero panel when no one steers it.
      const k = reduceMotion ? 0.6 : t * 0.22;
      target.set(Math.sin(k) * 0.75, 0.12 + Math.sin(k * 1.7) * 0.28, 0.38).applyMatrix4(hero.matrixWorld);
      current.lerp(target, reduceMotion ? 1 : 1 - Math.exp(-dt * 2.5));
    } else {
      raycaster.setFromCamera(ndc, camera);
      if (raycaster.ray.intersectPlane(lightPlane, target)) {
        target.clampLength(0, 6);
        current.lerp(target, reduceMotion ? 1 : 1 - Math.exp(-dt * 7));
      }
    }
    sun.position.copy(current);
    // A breath of flicker keeps the light feeling alive.
    sun.intensity = SUN * (1 + (reduceMotion ? 0 : Math.sin(t * 2.3) * 0.015));

    soil.value += (soil.target - soil.value) * (1 - Math.exp(-dt * 4));
    kit.setSoiling(soil.value);

    composer.render(dt);
    if (!ready) { ready = true; onReady && onReady(); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  return {
    setSoiling(level) { soil.target = level; },
  };
}
