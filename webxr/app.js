import * as THREE from "three";
import { VRButton } from "three/addons/webxr/VRButton.js";
import { ARButton } from "three/addons/webxr/ARButton.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { XRHandModelFactory } from "three/addons/webxr/XRHandModelFactory.js";

const statusEl = document.getElementById("status");
const setStatus = (msg, isErr = false) => {
  statusEl.textContent = msg;
  statusEl.className = isErr ? "err" : "";
};

// ── Viridis-ish colour ramp (year → colour) ─────────────────────────────────────
const VIRIDIS = [
  [0.267, 0.005, 0.329],
  [0.283, 0.141, 0.458],
  [0.254, 0.265, 0.53],
  [0.207, 0.372, 0.553],
  [0.164, 0.471, 0.558],
  [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518],
  [0.267, 0.749, 0.441],
  [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.15],
  [0.993, 0.906, 0.144],
];
function viridis(t) {
  t = Math.min(1, Math.max(0, t));
  const x = t * (VIRIDIS.length - 1);
  const i = Math.floor(x),
    f = x - i;
  const a = VIRIDIS[i],
    b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return new THREE.Color(
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  );
}

// Rounded-rect path on a 2-D context (avoids relying on ctx.roundRect support).
function _roundRectPath(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// Points tracing a rounded rectangle (centred at origin, XY plane) for a LineLoop.
function _roundedRectPoints(w, h, r, seg = 6) {
  const hw = w / 2,
    hh = h / 2;
  r = Math.min(r, hw, hh);
  const pts = [];
  const arc = (cx, cy, a0, a1) => {
    for (let i = 0; i <= seg; i++) {
      const a = a0 + (a1 - a0) * (i / seg);
      pts.push(
        new THREE.Vector3(cx + Math.cos(a) * r, cy + Math.sin(a) * r, 0),
      );
    }
  };
  arc(hw - r, hh - r, 0, Math.PI / 2); // top-right
  arc(-hw + r, hh - r, Math.PI / 2, Math.PI); // top-left
  arc(-hw + r, -hh + r, Math.PI, 1.5 * Math.PI); // bottom-left
  arc(hw - r, -hh + r, 1.5 * Math.PI, 2 * Math.PI); // bottom-right
  return pts;
}

// ── Text sprite helper ──────────────────────────────────────────────────────────
function makeLabel(text, color = "#dfe4ff", scale = 0.16) {
  const pad = 24,
    font = 64;
  const cv = document.createElement("canvas");
  const ctx = cv.getContext("2d");
  ctx.font = `bold ${font}px system-ui, sans-serif`;
  cv.width = ctx.measureText(text).width + pad * 2;
  cv.height = font + pad * 2;
  ctx.font = `bold ${font}px system-ui, sans-serif`;
  ctx.fillStyle = "rgba(8,10,18,0.72)";
  _roundRectPath(ctx, 0, 0, cv.width, cv.height, Math.min(cv.height * 0.4, 26));
  ctx.fill();
  ctx.fillStyle = color;
  ctx.textBaseline = "middle";
  ctx.fillText(text, pad, cv.height / 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  const spr = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, depthWrite: false }),
  );
  spr.scale.set((cv.width / cv.height) * scale, scale, 1);
  return spr;
}

// ── In-scene notice (DOM #status is invisible inside an immersive session) ──────
const _scratchA = new THREE.Vector3();
const _scratchB = new THREE.Vector3();
const _scratchC = new THREE.Vector3();
const _scratchQ = new THREE.Quaternion();
const _cardRaycaster = new THREE.Raycaster();
_cardRaycaster.near = 0;
_cardRaycaster.far = 5;

let _notice = null;
let _noticeExpiry = 0;
function showNotice(text, isErr = false, ttlMs = 4000) {
  setStatus(text, isErr); // mirror to the DOM (desktop)
  if (_notice) {
    scene.remove(_notice);
    _notice.material.map.dispose();
    _notice.material.dispose();
  }
  _notice = makeLabel(text, isErr ? "#ff6b6b" : "#dfe4ff", 0.085);
  _notice.renderOrder = 1000;
  scene.add(_notice);
  _noticeExpiry = ttlMs > 0 ? performance.now() + ttlMs : Infinity;
}

// ── Scene setup ─────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05060a);
scene.fog = new THREE.FogExp2(0x05060a, 0.12);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.05,
  100,
);
camera.position.set(0, 1.6, 2.6);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;

document.body.appendChild(renderer.domElement);

// Prefer immersive-ar: a dom-overlay session keeps the controls panel (incl. the
// 📷 file picker) usable *inside* the headset and floats the manifold in the real
// room. Fall back to the dark-space VR scene on headsets without passthrough.
// (Capturing a *new* real-world photo still needs WebXR camera access, which the
// Quest Browser lacks — you pick an image already on the headset.)
const CAM_CAP = { arSupported: false };
function _mountXrButton(useAR) {
  const btn = useAR
    ? ARButton.createButton(renderer, {
        optionalFeatures: ["dom-overlay", "local-floor", "hand-tracking"],
        domOverlay: { root: document.getElementById("overlay") },
      })
    : VRButton.createButton(renderer, {
        optionalFeatures: ["local-floor", "bounded-floor", "hand-tracking"],
      });
  btn.style.position = "static";
  btn.style.transform = "none";
  document.getElementById("xr-slot").appendChild(btn);
}
(navigator.xr?.isSessionSupported?.("immersive-ar") ?? Promise.resolve(false))
  .then((ar) => {
    CAM_CAP.arSupported = !!ar;
    _mountXrButton(!!ar);
  })
  .catch(() => _mountXrButton(false));

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(2, 4, 3);
scene.add(dir);

const grid = new THREE.GridHelper(10, 20, 0x223, 0x162033);
grid.material.transparent = true;
grid.material.opacity = 0.35;
scene.add(grid);

// Passthrough toggle: in an AR session (non-opaque blend) drop the dark
// backdrop/fog/grid so the real room shows through; restore for VR/desktop.
const _look = { bg: scene.background, fog: scene.fog };
let _trayReplaceAt = 0; // re-seat the shelf at the viewer
renderer.xr.addEventListener("sessionstart", () => {
  const s = renderer.xr.getSession();
  if (s && s.environmentBlendMode && s.environmentBlendMode !== "opaque") {
    scene.background = null;
    scene.fog = null;
    grid.visible = false;
    renderer.setClearAlpha(0);
  }
  _trayReplaceAt = performance.now() + 400; // after the XR pose is valid
});
renderer.xr.addEventListener("sessionend", () => {
  scene.background = _look.bg;
  scene.fog = _look.fog;
  grid.visible = true;
  renderer.setClearAlpha(1);
});

// Everything the user inspects lives in `world`, hovering at eye level.
const world = new THREE.Group();
world.position.set(0, 1.4, -1.0);
scene.add(world);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(world.position);
controls.enableDamping = true;
controls.update();

const rawGroup = new THREE.Group();
rawGroup.visible = false;
world.add(rawGroup);
const userGroup = new THREE.Group();
world.add(userGroup);

// Picking context, populated once /api/timeline resolves. The animation loop and
// the grip/key handlers all read this; `cloud` stays null while the fetch is in
// flight, so every reader must null-guard it.
const pick = { cloud: null, years: null, labels: null, yearRange: null };

// ── Build the manifold from the backend ─────────────────────────────────────────
function cloud(coords, years, [minY, maxY], size, opacity) {
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(coords.length * 3);
  const col = new Float32Array(coords.length * 3);
  for (let i = 0; i < coords.length; i++) {
    pos.set(coords[i], i * 3);
    const c = viridis((years[i] - minY) / Math.max(1, maxY - minY));
    col.set([c.r, c.g, c.b], i * 3);
  }
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return new THREE.Points(
    g,
    new THREE.PointsMaterial({
      size,
      vertexColors: true,
      transparent: true,
      opacity,
      sizeAttenuation: true,
      depthWrite: opacity > 0.6,
    }),
  );
}

// Per-cross-section sleeve radius from how far the raw embeddings stray from the
// time-axis there — the tube swells where the manifold fans out, narrows where
// it hugs the curve. Max-pooled then average-smoothed so the profile is smooth.
function curveRadiiFromSpread(curve, rawPoints, tubular) {
  const ns = curve.length;
  const raw = new Float32Array(ns);
  if (rawPoints && rawPoints.length) {
    for (const p of rawPoints) {
      let best = Infinity,
        bi = 0;
      for (let i = 0; i < ns; i++) {
        const dx = curve[i][0] - p[0];
        const dy = curve[i][1] - p[1];
        const dz = curve[i][2] - p[2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best) {
          best = d2;
          bi = i;
        }
      }
      const d = Math.sqrt(best);
      if (d > raw[bi]) raw[bi] = d;
    }
  }
  const pool = new Float32Array(ns),
    W = 14;
  for (let i = 0; i < ns; i++) {
    let m = 0;
    for (let k = -W; k <= W; k++) {
      const j = i + k;
      if (j >= 0 && j < ns && raw[j] > m) m = raw[j];
    }
    pool[i] = m;
  }
  const sm = new Float32Array(ns),
    S = 22;
  for (let i = 0; i < ns; i++) {
    let s = 0,
      c = 0;
    for (let k = -S; k <= S; k++) {
      const j = i + k;
      if (j >= 0 && j < ns) {
        s += pool[j];
        c++;
      }
    }
    sm[i] = s / c;
  }
  const MINR = 0.02,
    PAD = 0.02,
    MAXR = 0.45;
  const radii = new Float32Array(tubular + 1);
  for (let i = 0; i <= tubular; i++) {
    const ci = Math.round((i / tubular) * (ns - 1));
    radii[i] = Math.min(MAXR, Math.max(MINR, sm[ci] + PAD));
  }
  return radii;
}

// TubeGeometry with a per-cross-section radius (three's is fixed-radius).
function variableTube(path, radii, tubular, radial) {
  const frames = path.computeFrenetFrames(tubular, false);
  const P = new THREE.Vector3();
  const pos = new Float32Array((tubular + 1) * (radial + 1) * 3);
  let o = 0;
  for (let i = 0; i <= tubular; i++) {
    path.getPointAt(i / tubular, P);
    const N = frames.normals[i],
      B = frames.binormals[i],
      r = radii[i];
    for (let j = 0; j <= radial; j++) {
      const v = (j / radial) * Math.PI * 2;
      const sn = Math.sin(v),
        cs = -Math.cos(v);
      pos[o++] = P.x + r * (cs * N.x + sn * B.x);
      pos[o++] = P.y + r * (cs * N.y + sn * B.y);
      pos[o++] = P.z + r * (cs * N.z + sn * B.z);
    }
  }
  const idx = [];
  for (let i = 1; i <= tubular; i++) {
    for (let j = 1; j <= radial; j++) {
      const a = (radial + 1) * (i - 1) + (j - 1);
      const b = (radial + 1) * i + (j - 1);
      const c = (radial + 1) * i + j;
      const d = (radial + 1) * (i - 1) + j;
      idx.push(a, b, d, b, c, d);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

async function loadTimeline() {
  setStatus("Loading temporal manifold…");
  const r = await fetch("/api/timeline?backbone=clip");
  if (!r.ok) throw new Error(`timeline ${r.status}`);
  const d = await r.json();

  const manifold = cloud(d.points, d.years, d.yearRange, 0.022, 0.95);
  world.add(manifold);
  pick.cloud = manifold;
  pick.years = d.years;
  pick.labels = d.labels;
  pick.yearRange = d.yearRange;
  buildLegend();
  rawGroup.add(cloud(d.rawPoints, d.years, d.yearRange, 0.014, 0.28));

  // Bézier time-axis, coloured along its length with the SAME viridis ramp as
  // the points/legend (curve param 0→1 ≈ earliest→latest year) instead of red.
  const curvePts = d.curve.map((p) => new THREE.Vector3(...p));
  const path = new THREE.CatmullRomCurve3(curvePts);
  const TUBULAR = 600,
    RADIAL = 8;
  const tubeGeo = new THREE.TubeGeometry(path, TUBULAR, 0.012, RADIAL, false);
  const tvc = tubeGeo.attributes.position.count;
  const tcol = new Float32Array(tvc * 3);
  for (let k = 0; k < tvc; k++) {
    const ring = Math.floor(k / (RADIAL + 1)); // verts are grouped per ring
    const c = viridis(ring / TUBULAR);
    tcol[k * 3] = c.r;
    tcol[k * 3 + 1] = c.g;
    tcol[k * 3 + 2] = c.b;
  }
  tubeGeo.setAttribute("color", new THREE.BufferAttribute(tcol, 3));
  const tube = new THREE.Mesh(
    tubeGeo,
    new THREE.MeshBasicMaterial({ vertexColors: true }),
  );
  world.add(tube);

  // Glow: a couple of larger, additive, same-coloured tube shells around the
  // core. Additive blending makes overlaps brighten (a bloom-like halo) and reads
  // as light over AR passthrough too. (Real post-process bloom can't be used —
  // EffectComposer doesn't compose with WebXR's stereo render loop.) Same
  // TUBULAR/RADIAL ⇒ identical vertex count, so the viridis colours are reused.
  for (const [radius, opacity] of [
    [0.03, 0.32],
    [0.055, 0.16],
  ]) {
    const gGeo = new THREE.TubeGeometry(path, TUBULAR, radius, RADIAL, false);
    gGeo.setAttribute("color", new THREE.BufferAttribute(tcol.slice(), 3));
    const glow = new THREE.Mesh(
      gGeo,
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    );
    glow.renderOrder = 1;
    world.add(glow);
  }

  // Translucent "glass" sleeve enclosing the cloud — radius varies along the
  // curve with the embeddings' local spread (wider where they fan out).
  const radii = curveRadiiFromSpread(d.curve, d.rawPoints, TUBULAR);
  const shell = new THREE.Mesh(
    variableTube(path, radii, TUBULAR, 16),
    new THREE.MeshBasicMaterial({
      color: 0x9fc4ff,
      transparent: true,
      opacity: 0.1,
      side: THREE.DoubleSide,
      depthWrite: false, // don't occlude the points
    }),
  );
  shell.renderOrder = 2;
  world.add(shell);

  // ~12 year labels evenly along the timeline, offset to the SIDE of the ribbon
  // (perpendicular to the local timeline tangent, in the horizontal plane) so the
  // thrown image cards — which park centred on the curve and face the viewer —
  // don't overlap them. Offset (0.26 m) exceeds the card half-width (CARD_MAX/2).
  const N = d.points.length;
  const steps = 12;
  const _tan = new THREE.Vector3();
  const _side = new THREE.Vector3();
  const _up = new THREE.Vector3(0, 1, 0);
  const SIDE_OFFSET = 0.26;
  for (let s = 0; s < steps; s++) {
    const i = Math.round((s / (steps - 1)) * (N - 1));
    const lbl = makeLabel(String(d.labels[i]), "#cdd5ff", 0.12);
    const p = d.points[i];
    const a = d.points[Math.max(0, i - 1)];
    const b = d.points[Math.min(N - 1, i + 1)];
    _tan.set(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    _side.crossVectors(_tan, _up); // horizontal, perpendicular to the timeline
    if (_side.lengthSq() < 1e-6) _side.set(1, 0, 0); // tangent ≈ vertical → fallback
    _side.normalize().multiplyScalar(SIDE_OFFSET);
    lbl.position.set(p[0] + _side.x, p[1] + _side.y + 0.05, p[2] + _side.z);
    world.add(lbl);
  }
  setStatus(`Loaded ${N} years · ${d.yearRange[0]}–${d.yearRange[1]}.`);
}

// ── Image prediction → drop a marker ────────────────────────────────────────────
const COLORS = [0xe63946, 0x2a9d8f, 0xe9c46a, 0x9b5de5, 0xf4a261, 0x4cc9f0];
let userCount = 0;

// ── Throwable photo cards: upload → grab → throw → predict → park ───────────────
const CARD_MAX = 0.28; // m, longest card side
const FRAME_GAP = 0.016; // m, gap between the photo edge and the white outline
const FRAME_GLOW = 0.06; // m, how far the soft bloom extends beyond the outline
const CARD_RADIUS = 0.001; // m, rounded-corner radius of the photo itself
const FRAME_RADIUS = CARD_RADIUS + FRAME_GAP; // concentric rounding for the frame

// Soft bloom texture: a white rectangle stroke with a heavy canvas blur, sized so
// the stroke lands where the crisp outline is (FRAME_GLOW in from each edge) and
// the blur radiates outward. Used on an additive plane behind the outline — a
// real soft glow without post-process bloom (which can't compose with WebXR's
// stereo render loop).
function _frameGlowTexture(gw, gh) {
  const LONG = 256;
  const aspect = gw / gh;
  const cw = aspect >= 1 ? LONG : Math.round(LONG * aspect);
  const ch = aspect >= 1 ? Math.round(LONG / aspect) : LONG;
  const cv = document.createElement("canvas");
  cv.width = cw;
  cv.height = ch;
  const ctx = cv.getContext("2d");
  const mx = cw * (FRAME_GLOW / gw);
  const my = ch * (FRAME_GLOW / gh);
  const minside = Math.min(cw, ch);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = Math.max(2, minside * 0.018);
  ctx.lineJoin = "round";
  ctx.shadowColor = "#ffffff";
  ctx.shadowBlur = minside * 0.13;
  const radPx = FRAME_RADIUS * (cw / gw);
  for (let i = 0; i < 2; i++) {
    _roundRectPath(ctx, mx, my, cw - 2 * mx, ch - 2 * my, radPx);
    ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// White rounded-rect on black, matching a w×h photo — used as the photo's
// alphaMap so its corners read as rounded (transparent outside the radius).
function _roundedMask(w, h) {
  const LONG = 256;
  const aspect = w / h;
  const cw = aspect >= 1 ? LONG : Math.round(LONG * aspect);
  const ch = aspect >= 1 ? Math.round(LONG / aspect) : LONG;
  const cv = document.createElement("canvas");
  cv.width = cw;
  cv.height = ch;
  const ctx = cv.getContext("2d");
  const radPx = CARD_RADIUS * (LONG / Math.max(w, h));
  ctx.fillStyle = "#ffffff";
  _roundRectPath(ctx, 0, 0, cw, ch, radPx);
  ctx.fill();
  const tex = new THREE.CanvasTexture(cv);
  return tex;
}

// (Re)build a card's frame as ONE crisp white outline (photo + FRAME_GAP) plus a
// soft additive bloom plane behind it. Reused on creation and on texture resize.
function _buildFrame(group, w, h) {
  for (const c of group.children) {
    c.geometry.dispose();
    if (c.material.map) c.material.map.dispose();
    c.material.dispose();
  }
  group.clear();

  const ow = w + FRAME_GAP * 2;
  const oh = h + FRAME_GAP * 2;
  const outline = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(
      _roundedRectPoints(ow, oh, FRAME_RADIUS),
    ),
    new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );

  const gw = ow + FRAME_GLOW * 2;
  const gh = oh + FRAME_GLOW * 2;
  const glow = new THREE.Mesh(
    new THREE.PlaneGeometry(gw, gh),
    new THREE.MeshBasicMaterial({
      map: _frameGlowTexture(gw, gh),
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  );
  glow.position.z = -0.0008; // just behind the photo/outline
  group.add(glow, outline);
}
const THROW_SPEED = 0.7; // m/s release threshold for a "throw"
const GRAVITY = -1.0; // m/s² (softened for a readable arc)
const FLIGHT_MIN = 0.5,
  FLIGHT_MAX = 1.3; // s, distance/speed-scaled flight time
const SPIN_DAMP = 1.6; // 1/s tumble decay
const MAX_SPIN = 12; // rad/s cap on release tumble
const _TRAY_OFFSET = new THREE.Vector3(0, -0.35, 0);
const _texLoader = new THREE.TextureLoader();
const photoCards = [];

// Flight scratch (the loop processes cards serially, so singletons are safe).
const _fVel = new THREE.Vector3();
const _fAxis = new THREE.Vector3();
const _fTarget = new THREE.Vector3();
const _fNeed = new THREE.Vector3();
const _fQuat = new THREE.Quaternion();
const _qa = new THREE.Quaternion();
const _qb = new THREE.Quaternion();
const _qd = new THREE.Quaternion();
const _vwPos = new THREE.Vector3();
const _vwQuat = new THREE.Quaternion();
const _orient = new THREE.Object3D();

// Release velocity from the trajectory samples — recency-weighted over the last
// few segments so a quick flick (peak speed at release) is captured, not averaged
// away across the whole grab.
function releaseVelocity(s, out) {
  out.set(0, 0, 0);
  if (s.length < 2) return out;
  let n = 0,
    wsum = 0;
  for (let i = s.length - 1; i >= 1 && n < 3; i--, n++) {
    const a = s[i - 1],
      b = s[i];
    const dt = Math.max(1e-3, (b.t - a.t) / 1000);
    const w = n === 0 ? 2 : 1;
    out.x += (w * (b.x - a.x)) / dt;
    out.y += (w * (b.y - a.y)) / dt;
    out.z += (w * (b.z - a.z)) / dt;
    wsum += w;
  }
  if (wsum > 0) out.multiplyScalar(1 / wsum);
  return out;
}

// World pose of whichever camera is active (XR rig while presenting, else the
// desktop PerspectiveCamera).
function viewerPose(outPos, outQuat) {
  const cam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;
  cam.getWorldPosition(outPos);
  cam.getWorldQuaternion(outQuat);
}

// Park a card in a row on a "tray" ~0.6 m in front of and below the viewer,
// facing them. Card stays a child of `scene` so the auto-spin/orbit don't drag it.
function placeOnTray(card) {
  viewerPose(_scratchA, _scratchQ);
  const fwd = _scratchB.set(0, 0, -1).applyQuaternion(_scratchQ);
  fwd.y = 0;
  if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
  fwd.normalize();
  const right = _scratchC.set(-fwd.z, 0, fwd.x); // cross(fwd, up)
  const slot = card.userData.slot ?? 0; // stable; centred on the 4 examples
  card.position
    .copy(_scratchA)
    .addScaledVector(fwd, 0.6)
    .addScaledVector(right, (slot - 1.5) * 0.32)
    .add(_TRAY_OFFSET);
  card.lookAt(_scratchA);
  card.userData.state = "tray";
  card.userData.grabbedBy = null;
}

let _cardSlot = 0;

function _applyCardTexture(card, photo, frame, texUrl, revoke) {
  _texLoader.load(
    texUrl,
    (t) => {
      if (revoke) URL.revokeObjectURL(texUrl);
      t.colorSpace = THREE.SRGBColorSpace;
      card.userData.tex = t;
      photo.material.map = t;
      photo.material.color.set(0xffffff);
      photo.material.needsUpdate = true;
      const a = (t.image && t.image.width / t.image.height) || 1;
      let w = CARD_MAX,
        h = CARD_MAX;
      if (a >= 1) h = CARD_MAX / a;
      else w = CARD_MAX * a;
      photo.geometry.dispose();
      photo.geometry = new THREE.PlaneGeometry(w, h);
      if (photo.material.alphaMap) photo.material.alphaMap.dispose();
      photo.material.alphaMap = _roundedMask(w, h);
      photo.material.needsUpdate = true;
      _buildFrame(frame, w, h);
    },
    undefined,
    () => {
      if (revoke) URL.revokeObjectURL(texUrl);
      showNotice("Image load failed", true);
    },
  );
}

// src = { kind:"file", file, name }  or  { kind:"url", url, name }
function _makeCard(src) {
  const card = new THREE.Group();
  const photo = new THREE.Mesh(
    new THREE.PlaneGeometry(CARD_MAX, CARD_MAX),
    new THREE.MeshBasicMaterial({
      color: 0x333333,
      side: THREE.DoubleSide,
      transparent: true, // rounded corners come from the alphaMap set on load
    }),
  );
  // Glowing white outline standing off the photo by FRAME_GAP, with a soft bloom.
  const frame = new THREE.Group();
  _buildFrame(frame, CARD_MAX, CARD_MAX);
  card.add(photo, frame);
  card.userData = {
    src,
    photo,
    frame,
    tex: null,
    label: null,
    slot: _cardSlot++,
    state: "tray",
    grabbedBy: null,
    samples: [],
    flight: null,
  };
  photoCards.push(card);
  scene.add(card);
  placeOnTray(card);
  return { card, photo, frame };
}

function spawnPhotoCard(file) {
  const { card, photo, frame } = _makeCard({
    kind: "file",
    file,
    name: file.name,
  });
  _applyCardTexture(card, photo, frame, URL.createObjectURL(file), true);
  return card;
}

function spawnExampleCard(url, name) {
  const { card, photo, frame } = _makeCard({ kind: "url", url, name });
  _applyCardTexture(card, photo, frame, url, false);
  return card;
}

// Resolve the controller group from a raycast hit on a card's child mesh.
function _cardOf(obj) {
  while (obj && !obj.userData.src) obj = obj.parent;
  return obj || null;
}

// Card under a controller's ray, or within reach — only tray-state cards.
function pickCardForController(c) {
  c.updateWorldMatrix(true, false);
  _scratchQ.setFromRotationMatrix(c.matrixWorld);
  _cardRaycaster.ray.origin.setFromMatrixPosition(c.matrixWorld);
  _cardRaycaster.ray.direction
    .set(0, 0, -1)
    .applyQuaternion(_scratchQ)
    .normalize();
  const tray = photoCards.filter((k) => k.userData.state === "tray");
  if (tray.length === 0) return null;
  const hit = _cardRaycaster.intersectObjects(tray, true)[0];
  if (hit) {
    const k = _cardOf(hit.object);
    if (k) return k;
  }
  c.getWorldPosition(_scratchA);
  let best = null,
    bestD = 0.12 * 0.12; // 12 cm grab radius
  for (const k of tray) {
    const d = k.getWorldPosition(_scratchB).distanceToSquared(_scratchA);
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  return best;
}

// Derive real release velocity + tumble from the trajectory; throw or drop.
function releaseCard(card, c) {
  const s = card.userData.samples;
  releaseVelocity(s, _fVel); // world-space linear velocity
  _fAxis.set(0, 1, 0);
  let rate = 0;
  if (s.length >= 2) {
    const a = s[s.length - 2],
      b = s[s.length - 1];
    const dt = Math.max(1e-3, (b.t - a.t) / 1000);
    _qa.set(a.qx, a.qy, a.qz, a.qw);
    _qb.set(b.qx, b.qy, b.qz, b.qw);
    _qd.copy(_qb).multiply(_qa.invert()).normalize();
    const w = Math.min(1, Math.abs(_qd.w));
    const ssin = Math.sqrt(Math.max(1e-9, 1 - w * w));
    const sgn = _qd.w < 0 ? -1 : 1;
    _fAxis.set(_qd.x, _qd.y, _qd.z).multiplyScalar(sgn / ssin);
    if (!isFinite(_fAxis.x) || _fAxis.lengthSq() < 1e-6) _fAxis.set(0, 1, 0);
    _fAxis.normalize();
    rate = Math.min(MAX_SPIN, (2 * Math.acos(w)) / dt);
  }
  const speed = _fVel.length();
  scene.attach(card); // detach controller, keep transform
  card.userData.grabbedBy = null;
  if (speed >= THROW_SPEED) throwCard(card, _fVel, _fAxis, rate);
  else placeOnTray(card);
}

// Retry transient transport failures (the demo tunnel drops requests). Network
// errors and 5xx are retried with backoff; 4xx is returned as-is (real error).
async function _fetchRetry(url, opts, tries = 3) {
  let err;
  for (let i = 0; i < tries; i++) {
    try {
      const r = await fetch(url, opts);
      if (r.ok || (r.status >= 400 && r.status < 500)) return r;
      err = new Error(`HTTP ${r.status}`);
    } catch (e) {
      err = e;
    }
    await new Promise((res) => setTimeout(res, 250 * (i + 1)));
  }
  throw err;
}

// Launch the card on a real ballistic trajectory NOW, and resolve the year
// prediction in parallel. updateCards integrates gravity/tumble; once `pred`
// lands it guides the arc onto the predicted position. A no-velocity call
// (desktop key) synthesises a forward toss.
function throwCard(card, vel, axis, rate) {
  card.userData.state = "flying";
  const pos = card.getWorldPosition(new THREE.Vector3());
  if (!vel || vel.lengthSq() < 1e-6) {
    viewerPose(_vwPos, _vwQuat);
    vel = new THREE.Vector3(0, 0, -1)
      .applyQuaternion(_vwQuat)
      .multiplyScalar(2.4);
    vel.y += 0.6; // slight upward lob
    axis = new THREE.Vector3(1, 0, 0);
    rate = 4;
  }
  const f = {
    pos,
    vel: vel.clone(),
    spinAxis: (axis || new THREE.Vector3(0, 1, 0)).clone().normalize(),
    spinRate: Math.min(rate || 0, MAX_SPIN),
    t: 0,
    last: performance.now(),
    pred: null,
    gStart: null,
    dead: false,
  };
  card.userData.flight = f;

  (async () => {
    try {
      const fd = new FormData();
      const src = card.userData.src;
      if (src.kind === "file") {
        fd.append("file", src.file);
      } else {
        const ir = await _fetchRetry(src.url, undefined, 3);
        fd.append("file", await ir.blob(), src.name || "image.jpg");
      }
      const r = await _fetchRetry(
        "/api/predict",
        { method: "POST", body: fd },
        3,
      );
      if (!r.ok) {
        throw new Error(
          (await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`,
        );
      }
      const d = await r.json();
      if (card.userData.flight !== f) return; // superseded/parked
      f.pred = d; // glide is started & timed from HERE in updateCards, so
      // network latency can never make it teleport.
    } catch (err) {
      showNotice(`Prediction failed: ${err.message}`, true);
      f.dead = true;
    }
  })();
}

// Snap the card to its predicted local position under userGroup (child of
// world) so it survives grab/move/scale — same anchoring as markers/pins.
function parkCard(card) {
  const d = card.userData.flight.pred;
  world.updateWorldMatrix(true, false);
  userGroup.attach(card);
  card.position.set(...d.position);
  viewerPose(_scratchA, _scratchQ);
  card.lookAt(_scratchA);
  const lbl = makeLabel(`~${d.year}`, "#ffffff", 0.13);
  lbl.position.set(...d.position);
  lbl.position.y += 0.08;
  userGroup.add(lbl);
  card.userData.label = lbl;
  card.userData.state = "parked";
  card.userData.flight = null;
  showNotice(`~${d.year}`, false, 1800);
  predictions.push({ name: d.name, year: d.year });
  renderPredictions();
}

// Per-frame: sample grabbed cards' velocity, face tray cards at the viewer,
// advance the flight arc, and keep the notice in front of the viewer.
function updateCards(now) {
  if (_trayReplaceAt && now >= _trayReplaceAt) {
    _trayReplaceAt = 0;
    for (const c of photoCards) {
      if (c.userData.state === "tray") placeOnTray(c);
    }
  }
  for (const card of photoCards) {
    const u = card.userData;
    if (u.state === "grabbed") {
      card.getWorldPosition(_scratchA);
      card.getWorldQuaternion(_scratchQ);
      u.samples.push({
        x: _scratchA.x,
        y: _scratchA.y,
        z: _scratchA.z,
        qx: _scratchQ.x,
        qy: _scratchQ.y,
        qz: _scratchQ.z,
        qw: _scratchQ.w,
        t: now,
      });
      if (u.samples.length > 6) u.samples.shift();
    } else if (u.state === "tray") {
      viewerPose(_scratchA, _scratchQ);
      card.lookAt(_scratchA);
    } else if (u.state === "parked") {
      // Billboard the parked card at the viewer every frame (like the year-label
      // sprites) so it tracks you as you orbit/move, instead of facing only the
      // pose it had when it landed. lookAt works in world space and accounts for
      // userGroup/world's transform, so the timeline anchor is unaffected.
      viewerPose(_scratchA, _scratchQ);
      card.lookAt(_scratchA);
    } else if (u.state === "flying" && u.flight) {
      const f = u.flight;
      const dt = Math.min(0.05, Math.max(0, (now - f.last) / 1000));
      f.last = now;
      f.t += dt;

      let done = false;
      let gp = 0;

      if (!f.pred) {
        // Awaiting the year: a brief realistic toss that quickly damps to a
        // gentle hover, so it stays in view ("estimating") and never flies off.
        f.vel.multiplyScalar(Math.exp(-3.0 * dt));
        f.vel.y += GRAVITY * 0.3 * dt;
        f.pos.addScaledVector(f.vel, dt);
        if (f.dead || f.t > 20) {
          // real failure (after retries)
          card.userData.flight = null;
          placeOnTray(card);
          done = true;
        }
      } else {
        // Prediction in: a smooth, fixed-duration eased glide from wherever the
        // card is NOW to the timeline spot. Timed from arrival (gStart), so a
        // slow tunnel can never make it teleport. Target re-read each frame so
        // it stays correct if the manifold is moved/scaled mid-glide.
        world.updateWorldMatrix(true, false);
        _fTarget.set(...f.pred.position).applyMatrix4(world.matrixWorld);
        if (f.gStart == null) {
          f.gStart = f.t;
          f.gFrom = f.pos.clone();
          const dist = f.gFrom.distanceTo(_fTarget);
          f.gDur = THREE.MathUtils.clamp(dist / 1.6, FLIGHT_MIN, FLIGHT_MAX);
          f.gArc = Math.min(0.25, dist * 0.18);
        }
        gp = Math.min(1, (f.t - f.gStart) / Math.max(f.gDur, 1e-3));
        const e = gp < 0.5 ? 2 * gp * gp : 1 - Math.pow(-2 * gp + 2, 2) / 2;
        f.pos.lerpVectors(f.gFrom, _fTarget, e);
        f.pos.y += Math.sin(e * Math.PI) * f.gArc; // gentle arc
        if (gp >= 1) {
          card.position.copy(_fTarget);
          parkCard(card);
          done = true;
        }
      }

      if (!done) {
        card.position.copy(f.pos);
        if (f.spinRate > 0.01) {
          // damped tumble
          _fQuat.setFromAxisAngle(f.spinAxis, f.spinRate * dt);
          card.quaternion.premultiply(_fQuat);
          f.spinRate *= Math.exp(-SPIN_DAMP * dt);
        }
        if (f.pred && gp > 0.5) {
          // settle to a readable face over the back half of the glide
          viewerPose(_vwPos, _vwQuat);
          _orient.position.copy(f.pos);
          _orient.lookAt(_vwPos);
          card.quaternion.slerp(_orient.quaternion, ((gp - 0.5) / 0.5) * 0.35);
        }
      }
    }
  }
  if (_notice) {
    if (performance.now() > _noticeExpiry) {
      scene.remove(_notice);
      _notice.material.map.dispose();
      _notice.material.dispose();
      _notice = null;
    } else {
      viewerPose(_scratchA, _scratchQ);
      const fwd = _scratchB.set(0, 0, -1).applyQuaternion(_scratchQ);
      _notice.position.copy(_scratchA).addScaledVector(fwd, 1.2);
      _notice.position.y -= 0.15;
    }
  }
}

// Capture button (left X / buttons[4]). WebXR Raw Camera Access is gated behind
// `camera-access`; absent on Quest today, so this only explains the limitation.
const _btnPrev = new WeakMap();
function pollCaptureButton() {
  const session = renderer.xr.getSession();
  if (!session) return;
  for (const src of session.inputSources) {
    const gp = src.gamepad;
    if (!gp || src.handedness !== "left") continue;
    const btn = gp.buttons[4];
    if (!btn) continue;
    const prev = _btnPrev.get(src) || false;
    if (btn.pressed && !prev) {
      const camOK =
        !!session.enabledFeatures &&
        session.enabledFeatures.includes("camera-access");
      // DEFERRED: with camera-access, grab the per-eye camera image here, turn
      // it into a File, and call spawnPhotoCard(file) — identical downstream.
      showNotice(
        camOK
          ? "Camera capture not wired in this build."
          : "In-headset capture unavailable — upload an image instead.",
        true,
      );
    }
    _btnPrev.set(src, btn.pressed);
  }
}

document.getElementById("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  spawnPhotoCard(f); // prediction now happens on the throw
  showNotice(`“${f.name}” ready — throw it at the graph.`, false, 3000);
  e.target.value = "";
});

document.getElementById("raw").addEventListener("change", (e) => {
  rawGroup.visible = e.target.checked;
});

// ── Viridis legend + predicted-image list (toggled from the overlay) ────────────
const predictions = [];

function _yearColor(year) {
  const yr = pick.yearRange;
  const y = parseInt(year, 10);
  if (!yr || !Number.isFinite(y)) return "#888";
  const t = (y - yr[0]) / Math.max(1, yr[1] - yr[0]);
  return viridis(t).getStyle(); // same ramp as the point cloud
}

function buildLegend() {
  const stops = [];
  for (let i = 0; i <= 10; i++) {
    stops.push(`${viridis(i / 10).getStyle()} ${i * 10}%`);
  }
  document.getElementById("legend-bar").style.background =
    `linear-gradient(to right, ${stops.join(", ")})`;
  const yr = pick.yearRange;
  if (yr) {
    document.getElementById("legend-min").textContent = yr[0];
    document.getElementById("legend-mid").textContent = Math.round(
      (yr[0] + yr[1]) / 2,
    );
    document.getElementById("legend-max").textContent = yr[1];
  }
}

function renderPredictions() {
  const ul = document.getElementById("pred-list");
  if (predictions.length === 0) {
    ul.innerHTML =
      '<li class="muted">none yet — throw an image at the graph</li>';
    return;
  }
  ul.replaceChildren(
    ...predictions.map((p) => {
      const li = document.createElement("li");
      const sw = document.createElement("span");
      sw.className = "sw";
      sw.style.background = _yearColor(p.year);
      const nm = document.createElement("span");
      nm.textContent = p.name;
      const yr = document.createElement("span");
      yr.className = "yr";
      yr.textContent = `~${p.year}`;
      li.append(sw, nm, yr);
      return li;
    }),
  );
}

document.getElementById("legend-btn").addEventListener("click", () => {
  const el = document.getElementById("legend");
  el.hidden = !el.hidden;
});

// "📱 Phone cam": reveal the /phone URL + a QR to scan. QR lib is imported lazily
// (only when first opened) so it never affects initial load; if it fails the URL
// text alone is enough to type in.
document.getElementById("phone-btn").addEventListener("click", async () => {
  const panel = document.getElementById("phone-panel");
  panel.hidden = !panel.hidden;
  if (panel.hidden) return;
  const url = location.origin + "/phone";
  document.getElementById("phone-url").textContent = url;
  if (panel.dataset.qr) return; // already rendered
  try {
    const { default: QRCode } =
      await import("https://cdn.jsdelivr.net/npm/qrcode@1.5.3/+esm");
    await QRCode.toCanvas(document.getElementById("phone-qr"), url, {
      width: 180,
      margin: 1,
    });
    panel.dataset.qr = "1";
  } catch {
    document.getElementById("phone-qr").hidden = true; // fall back to URL text
  }
});

// ── VR controllers: grab to move/rotate, right thumbstick to scale ──────────────
let grabbedBy = null;
for (const i of [0, 1]) {
  const c = renderer.xr.getController(i);
  c.addEventListener("selectstart", () => {
    if (twoHandScaling) return; // mid two-handed zoom — don't grab
    const card = pickCardForController(c); // card grab takes priority
    if (card && card.userData.state === "tray") {
      card.userData.state = "grabbed";
      card.userData.grabbedBy = c;
      card.userData.samples.length = 0;
      c.attach(card);
      return;
    }
    if (grabbedBy) return;
    grabbedBy = c;
    c.attach(world); // keeps world transform, parents it to controller
  });
  c.addEventListener("selectend", () => {
    const held = photoCards.find((k) => k.userData.grabbedBy === c);
    if (held) {
      releaseCard(held, c);
      return;
    }
    if (grabbedBy !== c) return;
    grabbedBy = null;
    scene.attach(world); // re-parent to scene, transform preserved
  });
  c.addEventListener("squeezestart", () => pinHover(i)); // grip = pin pointed point
  c.add(
    new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0, -1),
      ]),
      new THREE.LineBasicMaterial({ color: 0x7aa2ff }),
    ),
  );
  scene.add(c);
}

// ── Hand tracking: two-handed pinch to zoom the model ───────────────────────────
// Each hand's pinch is detected from the thumb-tip↔index-tip joint distance (with
// hysteresis to avoid flicker). When BOTH hands pinch, the live distance between
// the two pinch points drives world.scale relative to where the gesture started —
// spread hands apart to zoom in, bring them together to zoom out. This is the
// hand-tracking analogue of the right thumbstick scale (applyThumbstickScale).
const _handFactory = new XRHandModelFactory();
const hands = [];
for (const i of [0, 1]) {
  const h = renderer.xr.getHand(i);
  h.add(_handFactory.createHandModel(h, "spheres")); // built-in, no external asset
  scene.add(h);
  hands.push(h);
}

const PINCH_ON = 0.022; // m, thumb↔index gap to start a pinch
const PINCH_OFF = 0.04; // m, gap to release it (hysteresis)
const _pinchOn = [false, false];
const _ji = new THREE.Vector3();
const _jt = new THREE.Vector3();

// True while hand `i` is pinching; writes the pinch midpoint to `outMid`.
function pinchInfo(hand, i, outMid) {
  const tip = hand.joints?.["index-finger-tip"];
  const thumb = hand.joints?.["thumb-tip"];
  if (!tip || !thumb || !tip.visible || !thumb.visible) {
    _pinchOn[i] = false;
    return false;
  }
  tip.getWorldPosition(_ji);
  thumb.getWorldPosition(_jt);
  const d = _ji.distanceTo(_jt);
  if (_pinchOn[i]) {
    if (d > PINCH_OFF) _pinchOn[i] = false;
  } else if (d < PINCH_ON) {
    _pinchOn[i] = true;
  }
  if (_pinchOn[i]) outMid.copy(_ji).add(_jt).multiplyScalar(0.5);
  return _pinchOn[i];
}

const _ptL = new THREE.Vector3();
const _ptR = new THREE.Vector3();
const _scaleGesture = { active: false, startDist: 0, startScale: 1 };
// Set while a two-handed scale is in progress so the pinch's select events don't
// also grab/move the model (see the selectstart handler).
let twoHandScaling = false;

function applyHandPinchScale() {
  if (!renderer.xr.getSession()) return;
  const p0 = pinchInfo(hands[0], 0, _ptL);
  const p1 = pinchInfo(hands[1], 1, _ptR);
  if (p0 && p1) {
    const dist = _ptL.distanceTo(_ptR);
    if (!_scaleGesture.active) {
      _scaleGesture.active = true;
      _scaleGesture.startDist = dist;
      _scaleGesture.startScale = world.scale.x;
      twoHandScaling = true;
      if (grabbedBy) {
        // a pinch may have already grabbed the model — hand it back to the scene
        // so scaling doesn't fight a controller-attached translate.
        grabbedBy = null;
        scene.attach(world);
      }
    } else if (_scaleGesture.startDist > 1e-4) {
      const s = _scaleGesture.startScale * (dist / _scaleGesture.startDist);
      world.scale.setScalar(THREE.MathUtils.clamp(s, 0.2, 6));
    }
  } else if (_scaleGesture.active) {
    _scaleGesture.active = false;
    twoHandScaling = false;
  }
}

// ── Point-at-to-inspect + grip-to-pin ───────────────────────────────────────────
// Tolerance in the cloud's *local* point space (rendered point size is 0.022).
// Each frame the raycaster's world-space Points threshold is set to this × the
// cloud's world scale, so picking stays accurate across the 0.2–6 zoom range and
// whether `world` is parented to the scene or to a grabbing controller.
const PICK_LOCAL_RADIUS = 0.04;
// NOTE: in-headset *camera* capture (WebXR Raw Camera Access) is gated behind
// `camera-access` (see pollCaptureButton) and unavailable in the Quest Browser
// today, so the universal image source is upload → throwable photo card. No
// `preserveDrawingBuffer`: capture would use the raw-camera texture, not the
// framebuffer.

function makeHover() {
  const ring = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.02),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      depthTest: false,
    }),
  );
  ring.renderOrder = 999;
  ring.visible = false;
  world.add(ring); // child of world → shares the cloud's exact transform
  return { ring, label: null, index: -1 };
}

function hideHover(h) {
  h.ring.visible = false;
  if (h.label) h.label.visible = false;
  h.index = -1;
}

// Resolve a configured raycaster against the manifold; update the hover's ring +
// (lazily rebuilt) label and remember the picked data index. `_hits` is reused
// to avoid per-frame allocation — safe because the loop calls resolvers serially.
const _hits = [];
function resolveHover(raycaster, h) {
  if (!pick.cloud) {
    hideHover(h);
    return;
  }
  pick.cloud.updateWorldMatrix(true, false);
  raycaster.params.Points.threshold =
    PICK_LOCAL_RADIUS * pick.cloud.matrixWorld.getMaxScaleOnAxis();
  _hits.length = 0;
  raycaster.intersectObject(pick.cloud, false, _hits);
  if (_hits.length === 0) {
    hideHover(h);
    return;
  }

  const idx = _hits[0].index; // Points vertex index === data index (cloud() writes i*3)
  const p = pick.cloud.geometry.attributes.position;
  h.ring.visible = true;
  h.ring.position.set(p.getX(idx), p.getY(idx), p.getZ(idx));
  if (idx !== h.index) {
    // rebuild the label canvas only when it changes
    if (h.label) {
      world.remove(h.label);
      h.label.material.map.dispose();
      h.label.material.dispose();
    }
    h.label = makeLabel(String(pick.years[idx]), "#ffffff", 0.12);
    world.add(h.label);
    h.index = idx;
  }
  h.label.position.set(p.getX(idx), p.getY(idx) + 0.06, p.getZ(idx));
  h.label.visible = true;
}

// Drop a permanent dated marker at a data index. Shared by the XR grip handler
// and the desktop key — mirrors the image-prediction marker recipe so pins look
// identical. userGroup is a child of world, so pins survive grab/move/scale.
function pinIndex(idx) {
  if (idx < 0 || !pick.cloud) return; // nothing pointed → no-op
  const p = pick.cloud.geometry.attributes.position;
  const color = COLORS[userCount++ % COLORS.length];

  const marker = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.035),
    new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.5,
    }),
  );
  marker.position.set(p.getX(idx), p.getY(idx), p.getZ(idx));
  userGroup.add(marker);

  const card = makeLabel(
    `${pick.years[idx]} · ${pick.labels[idx]}`,
    "#ffffff",
    0.13,
  );
  card.position.set(p.getX(idx), p.getY(idx) + 0.08, p.getZ(idx));
  userGroup.add(card);

  setStatus(`Pinned ${pick.years[idx]}.`);
}

// Per-controller XR hover. The ray is built from the controller's own world
// matrix (controllers are children of `scene`), so it is unaffected by `world`
// being reparented to a controller during grab. Direction (0,0,-1) matches the
// visual blue line, so the highlight tracks exactly where the line points.
const HOVER = [makeHover(), makeHover()];
const _xrRaycasters = [new THREE.Raycaster(), new THREE.Raycaster()];
const _xrRot = new THREE.Matrix4();
for (const rc of _xrRaycasters) {
  rc.near = 0;
  rc.far = 30;
}

function updateHover(ci) {
  const c = renderer.xr.getController(ci);
  const rc = _xrRaycasters[ci];
  c.updateWorldMatrix(true, false);
  _xrRot.identity().extractRotation(c.matrixWorld);
  rc.ray.origin.setFromMatrixPosition(c.matrixWorld);
  rc.ray.direction.set(0, 0, -1).applyMatrix4(_xrRot).normalize();
  resolveHover(rc, HOVER[ci]);
}

function pinHover(ci) {
  pinIndex(HOVER[ci].index);
}

// Desktop fallback (no XR controllers): mouse points, "p" pins. Same resolve
// path as VR so the whole feature is testable without a headset.
const _deskRay = new THREE.Raycaster();
_deskRay.near = 0;
_deskRay.far = 100;
const _deskHover = makeHover();
const _mouseNDC = new THREE.Vector2();
let _mouseInside = false;

renderer.domElement.addEventListener("pointermove", (e) => {
  _mouseNDC.x = (e.clientX / window.innerWidth) * 2 - 1;
  _mouseNDC.y = -(e.clientY / window.innerHeight) * 2 + 1;
  _mouseInside = true;
});
renderer.domElement.addEventListener("pointerleave", () => {
  _mouseInside = false;
});
window.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === "p") pinIndex(_deskHover.index);
  if (k === "t") {
    // throw newest tray card
    const card = [...photoCards]
      .reverse()
      .find((c) => c.userData.state === "tray");
    if (card) throwCard(card);
  }
  if (k === "c")
    showNotice(
      "Capture needs a camera-access XR device — upload instead.",
      true,
    );
});

function updateDesktopHover() {
  if (!_mouseInside) {
    hideHover(_deskHover);
    return;
  }
  _deskRay.setFromCamera(_mouseNDC, camera);
  resolveHover(_deskRay, _deskHover);
}

function applyThumbstickScale() {
  const session = renderer.xr.getSession();
  if (!session) return;
  for (const src of session.inputSources) {
    if (src.handedness !== "right" || !src.gamepad) continue;
    const y = src.gamepad.axes[3] ?? 0; // thumbstick vertical
    if (Math.abs(y) > 0.15) {
      const s = world.scale.x * (1 - y * 0.02);
      world.scale.setScalar(THREE.MathUtils.clamp(s, 0.2, 6));
    }
  }
}

// ── Loop ────────────────────────────────────────────────────────────────────────
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  const now = performance.now();
  if (renderer.xr.isPresenting) {
    applyThumbstickScale();
    applyHandPinchScale();
    pollCaptureButton();
    updateHover(0);
    updateHover(1);
    hideHover(_deskHover); // drop the desktop highlight on entering VR
  } else {
    world.rotation.y += 0.0015; // gentle auto-spin on desktop
    controls.update();
    hideHover(HOVER[0]);
    hideHover(HOVER[1]);
    updateDesktopHover();
  }
  updateCards(now);
  renderer.render(scene, camera);
});

loadTimeline().catch((e) =>
  setStatus(
    `Could not load timeline: ${e.message}. Is the server running with embeddings pulled?`,
    true,
  ),
);

// ── Phone-shared photos: appear and auto-throw the instant they're sent ─────────
// A phone at /phone POSTs a snapshot to /api/share; the server pushes it down this
// SSE stream. We spawn the same kind of card as an upload, then immediately throw
// it so it flies into the timeline and predicts with no manual toss.
function receiveSharedImage(url, name) {
  const card = spawnExampleCard(url, name || "photo");
  throwCard(card);
  showNotice("📷 Received", false, 1500);
}

const _seenShareIds = new Set();
function connectShareStream() {
  let es;
  try {
    es = new EventSource("/api/share/stream");
  } catch {
    return; // no EventSource support → phone sharing simply unavailable
  }
  es.onmessage = (e) => {
    let d;
    try {
      d = JSON.parse(e.data);
    } catch {
      return;
    }
    // Dedup by id: the server replays the last event on (re)connect, and
    // EventSource reconnects on transient drops — so the same photo can arrive
    // more than once. Only throw it the first time we see it.
    if (!d || !d.url || _seenShareIds.has(d.id)) return;
    _seenShareIds.add(d.id);
    receiveSharedImage(d.url, d.name);
  };
  // EventSource auto-reconnects on transient drops; nothing to handle here.
}
connectShareStream();

// Bundled demo images → a grabbable shelf, so there is something to throw
// without picking a file (re-seated in front of you on entering the headset).
fetch("/api/examples")
  .then((r) => (r.ok ? r.json() : { examples: [] }))
  .then((d) =>
    (d.examples || []).forEach((n) =>
      spawnExampleCard(
        `/examples/${encodeURIComponent(n)}`,
        n.replace(/\.[a-z0-9]+$/i, ""),
      ),
    ),
  )
  .catch(() => {});
