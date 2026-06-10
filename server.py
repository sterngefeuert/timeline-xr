"""WebXR backend for Timeline-VLM.

Serves the temporal-manifold visualization as JSON for an immersive Three.js /
WebXR frontend, and exposes an image-prediction endpoint.

This is intentionally standalone (it does NOT import app.py): app.py builds a
full Gradio UI at import time and is the live Hugging Face *Gradio* Space, which
must keep working untouched. The ~30 lines of pipeline math below are duplicated
verbatim from app.py on purpose — keep the two in sync if the math changes.

Run:
    pip install -r requirements.txt -r requirements-webxr.txt
    git lfs pull            # REQUIRED: encodings/*.npy are LFS pointers otherwise
    python server.py        # http://localhost:7860
"""

import asyncio
import io
import json
import os
import threading
import time
import uuid

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sklearn.decomposition import KernelPCA
from transformers import CLIPModel, CLIPProcessor

ROOT = os.path.dirname(os.path.abspath(__file__))
WEBXR_DIR = os.path.join(ROOT, "webxr")
EXAMPLES_DIR = os.path.join(ROOT, "examples")

DATA_PATHS = {
    "clip": {"embeddings": "encodings/timeline_embeddings.npy",
             "labels":     "encodings/timeline_labels.txt"},
    "eva":  {"embeddings": "encodings/eva/eva_timeline_embeddings.npy",
             "labels":     "encodings/eva/eva_labels_timeline.txt"},
}

# ── Pipeline math (mirrors app.py) ──────────────────────────────────────────────
def _de_casteljau(pts, t):
    pts = pts.copy()
    for r in range(1, len(pts)):
        pts[:len(pts)-r] = (1-t)*pts[:len(pts)-r] + t*pts[1:len(pts)-r+1]
    return pts[0]


def bezier_curve(ctrl, n=1000):
    return np.array([_de_casteljau(ctrl, t) for t in np.linspace(0, 1, n)])


def project_onto_curve(points, curve):
    proj, idx = [], []
    for p in points:
        i = int(np.argmin(np.linalg.norm(curve - p, axis=1)))
        proj.append(curve[i]); idx.append(i)
    return np.array(proj), np.array(idx)


def build_kpca_and_curve(t_embs, t_lbls, n_ctrl):
    years = []
    for l in t_lbls:
        try:    years.append(int(l))
        except: years.append(1900)
    kpca     = KernelPCA(n_components=3, kernel="cosine")
    reduced  = kpca.fit_transform(t_embs)
    n_ctrl   = min(int(n_ctrl), len(reduced))
    ctrl_idx = np.linspace(0, len(reduced)-1, n_ctrl, dtype=int)
    curve    = bezier_curve(reduced[ctrl_idx], n=1000)
    proj_t, t_idx = project_onto_curve(reduced, curve)
    return kpca, reduced, curve, proj_t, t_idx, years


# ── Data loading with a clear LFS-pointer guard ─────────────────────────────────
def _load_embeddings(path):
    abspath = os.path.join(ROOT, path)
    if not os.path.exists(abspath):
        raise RuntimeError(f"Missing embeddings file: {path}")
    with open(abspath, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not real data. Run `git lfs pull` "
            "(install git-lfs first) before starting the server."
        )
    return np.load(abspath)


# ── Coordinate normalization ────────────────────────────────────────────────────
# KPCA output is tiny and off-origin; rescale the whole scene into a tidy cube so
# it's comfortable to inspect at room scale in VR. The SAME transform must be
# applied to predicted user points, so we store it per backbone.
CUBE_EDGE = 2.0  # metres, full extent of the longest axis


def _fit_normalizer(*point_sets):
    allp = np.vstack(point_sets)
    center = allp.mean(axis=0)
    extent = float(np.max(np.abs(allp - center))) or 1.0
    scale = (CUBE_EDGE / 2.0) / extent
    return center.astype(float), scale


def _apply_norm(pts, center, scale):
    return (np.asarray(pts, dtype=float) - center) * scale


# ── Model + per-backbone scene cache ────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading CLIP on {device}…", flush=True)
_HF_MODEL_ID = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(_HF_MODEL_ID)
clip_model = CLIPModel.from_pretrained(_HF_MODEL_ID).to(device)
clip_model.eval()
print("CLIP loaded.", flush=True)

_scene_cache = {}


def get_scene(backbone="clip", n_ctrl=300):
    """Build (and cache) the normalized timeline scene for a backbone."""
    key = (backbone, int(n_ctrl))
    if key in _scene_cache:
        return _scene_cache[key]
    if backbone not in DATA_PATHS:
        raise HTTPException(400, f"Unknown backbone '{backbone}'")

    t_embs = _load_embeddings(DATA_PATHS[backbone]["embeddings"])
    with open(os.path.join(ROOT, DATA_PATHS[backbone]["labels"])) as f:
        t_lbls = [l.strip() for l in f]

    kpca, reduced, curve, proj_t, t_idx, years = build_kpca_and_curve(
        t_embs, t_lbls, n_ctrl
    )
    center, scale = _fit_normalizer(reduced, curve)

    scene = {
        "kpca": kpca, "curve": curve, "t_idx": t_idx,
        "labels": t_lbls, "center": center, "scale": scale,
        "json": {
            "backbone": backbone,
            "points": _apply_norm(proj_t, center, scale).round(4).tolist(),
            "rawPoints": _apply_norm(reduced, center, scale).round(4).tolist(),
            "years": years,
            "labels": t_lbls,
            "curve": _apply_norm(curve, center, scale).round(4).tolist(),
            "yearRange": [min(years), max(years)],
        },
    }
    _scene_cache[key] = scene
    return scene


# ── API ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Timeline-VLM WebXR")


# The frontend has no build step / content hashing, so a long-lived cache makes
# headsets keep running stale JS across deploys (this bit the phone-share rollout:
# the Quest served a cached app.js without the SSE client). Force revalidation on
# the HTML/JS entrypoints — they're tiny, and ETags still yield 304s when unchanged.
@app.middleware("http")
async def _no_store_frontend(request: Request, call_next):
    resp = await call_next(request)
    path = request.url.path
    if path in ("/", "/phone") or path.endswith((".html", ".js")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/api/timeline")
def api_timeline(backbone: str = "clip", n_ctrl: int = 300):
    return get_scene(backbone, n_ctrl)["json"]


# Model + sklearn state is not thread-safe, so serialize inference. Defining the
# endpoint as a regular `def` (not `async`) makes Starlette run it in its
# threadpool, so this blocking work never freezes the event loop — keep-alives
# and a rapidly-thrown second image are still serviced instead of stalling.
_PREDICT_LOCK = threading.Lock()


@app.post("/api/predict")
def api_predict(file: UploadFile = File(...), backbone: str = "clip"):
    # CLIP ViT-B/32 is the only encoder we load, so prediction is only
    # coherent against the clip-space timeline (matches app.py's known quirk).
    if backbone != "clip":
        raise HTTPException(
            400, "Image prediction is only available with the 'clip' backbone."
        )
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image file.")

    with _PREDICT_LOCK:
        scene = get_scene(backbone)
        inputs = clip_processor(images=[img], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(
                pixel_values=pixel_values).cpu().numpy()

        r_user = scene["kpca"].transform(emb)
        p_user, u_idx = project_onto_curve(r_user, scene["curve"])
        near = int(np.argmin(np.abs(scene["t_idx"] - u_idx[0])))
        pred_year = scene["labels"][near]
        pos = _apply_norm(p_user[0], scene["center"], scene["scale"])

    return {
        "name": file.filename or "image",
        "year": pred_year,
        "position": [round(float(v), 4) for v in pos],
    }


@app.get("/")
def index():
    # No build step → no content-hashed asset names, and the reverse proxy slaps a
    # long max-age on *.js that overrides our no-cache header. So stamp app.js with
    # its mtime: the URL changes whenever the file does, forcing a fresh fetch on
    # the next deploy regardless of the cached copy. `/` itself is served no-cache
    # (see _no_store_frontend), so the headset always gets the current stamp.
    with open(os.path.join(WEBXR_DIR, "index.html")) as f:
        html = f.read()
    try:
        ver = int(os.path.getmtime(os.path.join(WEBXR_DIR, "app.js")))
    except OSError:
        ver = 0
    html = html.replace('src="./app.js"', f'src="./app.js?v={ver}"')
    return HTMLResponse(html)


@app.get("/api/examples")
def api_examples():
    """Bundled demo images, offered as a grabbable shelf inside the headset."""
    exts = (".jpg", ".jpeg", ".png", ".webp")
    try:
        names = sorted(f for f in os.listdir(EXAMPLES_DIR)
                        if f.lower().endswith(exts))
    except FileNotFoundError:
        names = []
    return {"examples": names}


# ── Phone → headset photo sharing ───────────────────────────────────────────────
# A phone opens GET /phone, snaps a picture in-browser, and POSTs it to
# /api/share. The headset page holds an SSE connection to /api/share/stream and
# spawns + auto-throws the photo the instant it arrives. Single global room (any
# phone → every connected headset); images are held in memory and served back by
# id. Kept deliberately tiny — no DB, no auth — for the homelab demo.
_subscribers: "set[asyncio.Queue]" = set()
_shared_images: "dict[str, dict]" = {}
_SHARED_MAX = 24  # cap retained images so memory can't grow unbounded
# A headset that connects (or reconnects) just after a phone send would otherwise
# miss it — SSE only reaches subscribers live at send time. So remember the last
# event and replay it to a freshly-connected stream within this window; the client
# dedups by id, so a keepalive reconnect never re-throws the same photo.
_last_event = None  # (event_dict, monotonic_ts)
_REPLAY_WINDOW = 90  # s


@app.get("/phone")
def phone():
    return FileResponse(os.path.join(WEBXR_DIR, "phone.html"))


@app.post("/api/share")
async def api_share(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    img_id = uuid.uuid4().hex
    _shared_images[img_id] = {
        "bytes": data,
        "type": file.content_type or "image/jpeg",
        "name": file.filename or "photo.jpg",
    }
    # Evict oldest beyond the cap (dicts keep insertion order).
    while len(_shared_images) > _SHARED_MAX:
        _shared_images.pop(next(iter(_shared_images)))
    event = {
        "id": img_id,
        "name": _shared_images[img_id]["name"],
        "url": f"/api/share/img/{img_id}",
    }
    global _last_event
    _last_event = (event, time.monotonic())
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # a wedged subscriber shouldn't block the sender
    return {"ok": True, "id": img_id}


@app.get("/api/share/img/{img_id}")
def api_share_img(img_id: str):
    rec = _shared_images.get(img_id)
    if not rec:
        raise HTTPException(404, "No such shared image.")
    return Response(content=rec["bytes"], media_type=rec["type"])


@app.get("/api/share/stream")
async def api_share_stream(request: Request):
    """Server-Sent Events: one message per shared photo, plus periodic
    keepalive comments so proxies/the browser hold the connection open."""
    q: "asyncio.Queue" = asyncio.Queue(maxsize=32)
    _subscribers.add(q)

    async def gen():
        try:
            yield ": connected\n\n"  # comment line — opens the stream immediately
            le = _last_event
            if le and (time.monotonic() - le[1]) < _REPLAY_WINDOW:
                yield f"data: {json.dumps(le[0])}\n\n"  # catch a just-missed send
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _subscribers.discard(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx, etc.)
        },
    )


if os.path.isdir(EXAMPLES_DIR):
    app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")
app.mount("/", StaticFiles(directory=WEBXR_DIR), name="webxr")


if __name__ == "__main__":
    # Warm the default scene so the first request is fast (and fails loudly
    # here, not mid-request, if embeddings are still LFS pointers).
    try:
        get_scene("clip")
        print("Timeline scene ready.", flush=True)
    except RuntimeError as e:
        print(f"\n  WARNING: {e}\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
