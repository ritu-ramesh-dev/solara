// Home-page hero: one datasheet-accurate solar module in darkness, lit by a
// small sun that follows the cursor. Drag to turn the panel in 3D.
import * as THREE from "three";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { createPanelKit } from "./panel.js";

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const finePointer = matchMedia("(pointer: fine)").matches;

const HOME_YAW = -0.5, HOME_PITCH = 0.0;

// Film grain and vignette, applied after tone mapping.
const FinishShader = {
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uGrain: { value: 0.045 }, uVignette: { value: 0.55 } },
  vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uTime, uGrain, uVignette; varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      float n = hash(floor(gl_FragCoord.xy) + fract(uTime) * 917.0) - 0.5;
      c.rgb += n * uGrain * (0.35 + 0.65 * (1.0 - c.g));
      vec2 d = (vUv - 0.5) * vec2(1.0, 1.18);
      c.rgb *= mix(1.0, smoothstep(0.92, 0.22, length(d)), uVignette);
      gl_FragColor = c;
    }`,
};

// Dust motes drifting in the air; they glow when they pass through the sunlight.
function makeMotes(count) {
  const pos = new Float32Array(count * 3), seed = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    pos[i * 3] = (Math.random() - 0.5) * 3.4;
    pos[i * 3 + 1] = (Math.random() - 0.5) * 3.4;
    pos[i * 3 + 2] = (Math.random() - 0.5) * 2.4 + 0.4;
    seed[i] = Math.random();
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("seed", new THREE.BufferAttribute(seed, 1));
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uSun: { value: new THREE.Vector3() }, uScale: { value: 1 } },
    vertexShader: `
      attribute float seed; uniform float uTime, uScale; uniform vec3 uSun; varying float vGlow;
      void main(){
        vec3 p = position;
        float t = uTime * (0.015 + seed * 0.02);
        p.y = mod(p.y + t + 1.7, 3.4) - 1.7;
        p.x += sin(uTime * 0.21 + seed * 40.0) * 0.06;
        p.z += cos(uTime * 0.17 + seed * 23.0) * 0.05;
        float d = distance(p, uSun);
        vGlow = 0.03 + 1.6 * exp(-d * d * 3.2);
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = uScale * (0.6 + seed * 1.4) * 14.0 / -mv.z;
      }`,
    fragmentShader: `
      varying float vGlow;
      void main(){
        float r = length(gl_PointCoord - 0.5);
        float a = smoothstep(0.5, 0.0, r) * vGlow;
        gl_FragColor = vec4(vec3(1.0, 0.86, 0.66) * a, a);
      }`,
  });
  return new THREE.Points(geo, mat);
}

export function startScene(canvas, { onReady, hdr = "assets/env/aarfontein_dusk_1k.hdr" } = {}) {
  const small = Math.min(innerWidth, innerHeight) < 700;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, small ? 1.5 : 1.75));
  renderer.setSize(innerWidth, innerHeight, false);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#040404");

  // Reflections: a dim studio until the real desert-dusk HDRI arrives.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.05;
  scene.environmentRotation = new THREE.Euler(0, 4.0, 0);
  new RGBELoader().load(hdr, (t) => {
    scene.environment = pmrem.fromEquirectangular(t).texture;
    scene.environmentIntensity = 0.5;
    t.dispose();
  });

  const camera = new THREE.PerspectiveCamera(30, innerWidth / innerHeight, 0.05, 60);

  // ---- the module ------------------------------------------------------------
  const kit = createPanelKit(renderer, { quality: small ? "low" : "high" });
  const turntable = new THREE.Group();          // yaw (drag left/right)
  const tilt = new THREE.Group();               // pitch (drag up/down)
  tilt.add(kit.panel);
  turntable.add(tilt);
  kit.panel.rotation.x = -0.12;                 // leans back a little, as on a mount
  scene.add(turntable);

  // An unseen floor catches the light the panel blocks, so it does not float.
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShadowMaterial({ opacity: 0.45 })); // taste-skill-ignore (a WebGL ground shadow, not a CSS shadow)
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -kit.height / 2 - 0.28;
  floor.receiveShadow = true;
  scene.add(floor);

  // ---- light -------------------------------------------------------------------
  const SUN = 3.2;
  const sun = new THREE.PointLight("#ffeccc", SUN, 4.5, 2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(small ? 512 : 1024, small ? 512 : 1024);
  sun.shadow.bias = -0.0006;
  sun.shadow.normalBias = 0.012;
  sun.shadow.radius = 5;
  const orb = new THREE.Mesh(new THREE.SphereGeometry(0.018, 32, 16),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(8, 7, 5.6), toneMapped: false }));
  sun.add(orb);
  scene.add(sun);
  // Cool rim light from behind draws the frame's silhouette.
  const rim = new THREE.DirectionalLight("#8fa6d8", 0.55);
  rim.position.set(-2.5, 2.2, -3.5);
  scene.add(rim);
  const kick = new THREE.DirectionalLight("#ffd9a8", 0.12);
  kick.position.set(3, -1, 2);
  scene.add(kick);

  const motes = makeMotes(small ? 320 : 700);
  scene.add(motes);

  // ---- post-processing -----------------------------------------------------------
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth / 2, innerHeight / 2), 0.6, 0.55, 1.6);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  const finish = new ShaderPass(FinishShader);
  composer.addPass(finish);

  // ---- framing ------------------------------------------------------------------
  let dist = 6;
  function layout() {
    const aspect = innerWidth / innerHeight;
    camera.aspect = aspect;
    // Panel height as a share of the view: smaller on phones, where the title,
    // subtitle and dust control need the lower half of the screen.
    const share = aspect < 1 ? 0.4 : 0.56;
    const fitH = kit.height / 0.536 / share;
    const fitW = (kit.height * 0.9) / 0.536 / aspect;  // room to turn on narrow screens
    dist = Math.max(fitH, fitW, 5.2);
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight, false);
    composer.setSize(innerWidth, innerHeight);
    bloom.setSize(innerWidth / 2, innerHeight / 2);
    motes.material.uniforms.uScale.value = renderer.getPixelRatio() * innerHeight / 900;
  }
  layout();
  addEventListener("resize", layout);

  // ---- interaction: drag to turn, hover to steer the sun ---------------------------
  const rot = { yaw: HOME_YAW - 1.1, pitch: -0.25, vYaw: 0, pitchTarget: HOME_PITCH };
  let dragging = false, lastX = 0, lastY = 0, lastInput = -1e9;
  const ndc = new THREE.Vector2(0.25, 0.2);
  let hovering = false;

  canvas.style.touchAction = "pan-y";
  canvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();           // no text selection while turning the panel
    window.getSelection()?.removeAllRanges();
    dragging = true; lastX = e.clientX; lastY = e.clientY; lastInput = performance.now();
    canvas.setPointerCapture(e.pointerId);
    document.body.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (e) => {
    if (e.pointerType === "mouse") {
      ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
      hovering = true;
    }
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    rot.yaw += dx * 0.006;
    rot.vYaw = dx * 0.006 * 60;
    if (e.pointerType !== "touch") rot.pitchTarget = THREE.MathUtils.clamp(rot.pitchTarget + dy * 0.004, -0.75, 0.6);
    lastInput = performance.now();
  });
  const release = (e) => {
    dragging = false;
    document.body.classList.remove("dragging");
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("pointerleave", () => { hovering = false; });
  canvas.addEventListener("dblclick", () => { rot.vYaw = 0; rot.pitchTarget = HOME_PITCH; rot.yaw = rot.yaw % (Math.PI * 2); rot.homing = true; });

  // ---- scroll: fade out as the page moves on --------------------------------------
  let visible = 1;
  const onScroll = () => {
    visible = THREE.MathUtils.clamp(1.3 - scrollY / innerHeight, 0, 1);
    canvas.style.opacity = String(visible);
  };
  addEventListener("scroll", onScroll, { passive: true });

  const soil = { value: 0, target: 0 };
  const raycaster = new THREE.Raycaster();
  const plane = new THREE.Plane();
  const target = new THREE.Vector3(), current = new THREE.Vector3(0.5, 0.6, 1.4), toCam = new THREE.Vector3();
  const clock = new THREE.Clock();
  const easeOut = (x) => 1 - Math.pow(1 - x, 3);
  const INTRO = reduceMotion ? 0 : 3.2;
  let ready = false;

  function frame() {
    requestAnimationFrame(frame);
    const dt = Math.min(clock.getDelta(), 0.05);
    const t = clock.elapsedTime;
    if (visible <= 0.001) return;   // nothing to draw once scrolled away

    // Opening shot: the camera rises and closes in while the panel turns to face us.
    const k = INTRO ? easeOut(Math.min(t / INTRO, 1)) : 1;
    const d = dist * THREE.MathUtils.lerp(1.45, 1, k);
    camera.position.set(Math.sin(0.08) * d, THREE.MathUtils.lerp(-0.9, 0.42, k), Math.cos(0.08) * d);
    camera.lookAt(0, THREE.MathUtils.lerp(-0.2, 0.08, k), 0);
    const lift = innerWidth / innerHeight > 1 ? 0.13 : 0.27;
    camera.setViewOffset(innerWidth, innerHeight, 0, innerHeight * lift, innerWidth, innerHeight);

    // Turning: inertia after a flick, slow drift when left alone, spring on pitch.
    const idle = performance.now() - lastInput > 6000;
    if (t < INTRO && !dragging) rot.yaw = THREE.MathUtils.lerp(HOME_YAW - 1.1, HOME_YAW, k);
    else if (!dragging) {
      rot.yaw += rot.vYaw * dt;
      rot.vYaw *= Math.exp(-dt * 2.8);
      if (rot.homing) {
        rot.yaw += (HOME_YAW - rot.yaw) * (1 - Math.exp(-dt * 3));
        if (Math.abs(HOME_YAW - rot.yaw) < 0.002) rot.homing = false;
      } else if (idle && !reduceMotion) rot.yaw += dt * 0.06;
    }
    rot.pitch += (rot.pitchTarget - rot.pitch) * (1 - Math.exp(-dt * (dragging ? 12 : 4)));
    if (!dragging && idle) rot.pitchTarget += (HOME_PITCH - rot.pitchTarget) * (1 - Math.exp(-dt * 0.8));
    turntable.rotation.y = rot.yaw;
    tilt.rotation.x = rot.pitch;

    // The sun lives on a plane between the camera and the panel.
    toCam.copy(camera.position).normalize();
    plane.setFromNormalAndCoplanarPoint(toCam, toCam.clone().multiplyScalar(1.05));
    if (finePointer && hovering && !dragging) {
      raycaster.setFromCamera(ndc, camera);
      if (raycaster.ray.intersectPlane(plane, target)) target.clampLength(0, 3);
    } else if (!dragging) {
      const a = reduceMotion ? 0.7 : t * 0.2;
      target.set(Math.sin(a) * 0.65, 0.35 + Math.sin(a * 1.6) * 0.45, 0).add(toCam.clone().multiplyScalar(1.05));
    }
    current.lerp(target, 1 - Math.exp(-dt * (hovering ? 6 : 1.6)));
    sun.position.copy(current);
    sun.intensity = SUN * (1 + (reduceMotion ? 0 : Math.sin(t * 2.1) * 0.015));

    soil.value += (soil.target - soil.value) * (1 - Math.exp(-dt * 3));
    kit.setSoiling(soil.value);

    motes.material.uniforms.uTime.value = t;
    motes.material.uniforms.uSun.value.copy(current);
    finish.uniforms.uTime.value = t;

    composer.render(dt);
    if (!ready) { ready = true; onReady && onReady(); }
  }
  requestAnimationFrame(frame);

  return {
    setSoiling(level) { soil.target = level; },
  };
}
