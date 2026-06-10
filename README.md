# Timeline-VLM WebXR

The demo (a) renders precomputed CLIP timeline embeddings (years **1700–2024**) as a
3-D point cloud threaded by a learned Bézier *time-axis*, explorable in VR or by
desktop orbit, and (b) estimates the year of an uploaded image by projecting its CLIP
embedding onto that same axis.

![description](demo.gif)




Everything needed to reproduce the demo is contained in this repository; no training
is required, as the timeline embeddings are precomputed and version-controlled.

---

## 1. System overview

| Component | Implementation | File(s) |
|-----------|----------------|---------|
| Backend | Standalone FastAPI server (~360 lines), serves scene JSON + image prediction | `server.py` |
| Frontend | Plain Three.js (CDN, no build step) + WebXR | `webxr/index.html`, `webxr/app.js`, `webxr/phone.html` |
| Data | Precomputed timeline embeddings + per-year labels (Git LFS) | `encodings/` |
| Demo images | Bundled example images for the in-headset shelf | `examples/` |
| Encoder | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), downloaded on first run | (HF Hub) |


---

## 2. Requirements

### Software
- **Python 3.11–3.13.** Local development was pinned to `3.11.9` (`.tool-versions`);
  the Docker image uses `python:3.13-slim`. Either works with the pinned wheels.
- **Git LFS** — required; embeddings and example images are LFS-tracked.
- Python dependencies (`requirements.txt`):
  ```
  torch==2.6.0
  torchvision==0.21.0
  transformers==4.47.0
  Pillow
  scikit-learn
  numpy
  fastapi
  uvicorn[standard]
  python-multipart
  ```

### Hardware
- **CPU is sufficient.** The model auto-selects CUDA if available
  (`server.py:120`), but inference is a single CLIP forward pass per image, so a
  GPU is not required to reproduce results.
- ~2 GB disk for the CLIP weights (cached by `transformers` on first run).
- For VR: any WebXR-capable headset (e.g. Meta Quest) with a browser. WebXR requires
  a secure context (`https://` or `localhost`).

### Determinism note
KernelPCA, the Bézier fit, and the projection are all deterministic given fixed
input embeddings — there is no random seed to set. Reproducing the visualization and
predictions therefore depends only on the LFS data + the CLIP weights resolving to
the same checkpoint (`openai/clip-vit-base-patch32`). Minor floating-point
differences across CPU/GPU/BLAS backends may shift a marker by a sub-millimetre but
do not change predicted years on the bundled examples.

---

## 3. Data

`encodings/` holds the precomputed timeline embeddings and their year labels. Each
`.npy` is one CLIP/EVA embedding per year; the parallel `.txt` lists one year per
line (verified equal row counts).

| Backbone | Embeddings | Shape / dtype | Labels | Rows |
|----------|------------|---------------|--------|------|
| `clip` | `encodings/timeline_embeddings.npy` | `(325, 512)` float32 | `encodings/timeline_labels.txt` | 325 |
| `eva` | `encodings/eva/eva_timeline_embeddings.npy` | `(324, 512)` float16 | `encodings/eva/eva_labels_timeline.txt` | 324 |

`DATA_PATHS` (`server.py:44`) maps each backbone to its `{embeddings, labels}` pair.

> **Provenance.** These embeddings are the precomputed temporal-manifold features
> from the upstream paper/demo; this repo redistributes them via Git LFS and does
> not regenerate them. Image *prediction* is **clip-only** — CLIP ViT-B/32 is the
> only encoder loaded, so `/api/predict` rejects `backbone != "clip"`
> (`server.py:198`). The `eva` backbone is for timeline visualization only.

## 4. Installation & run (local)

```bash
git clone <repo-url> timeline-xr && cd timeline-xr
git lfs install                     # one-time
git lfs pull                        # REQUIRED — resolves the .npy / .jpg blobs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # see torch note below
python server.py                    # serves on http://0.0.0.0:7860
```

Open **http://localhost:7860** — desktop works immediately: drag to orbit, scroll to
zoom, toggle raw embeddings, and upload an image to estimate its year.

On startup the server the default `clip` scene and prints `Timeline scene
ready.` (`server.py:359`). If the embeddings are still LFS pointers it instead prints
a `WARNING:` line telling you to run `git lfs pull`.

- **`PORT` override:** `PORT=8080 python server.py`.
- **torch note:** if the pinned `torch==2.6.0` / `torchvision==0.21.0` wheels are
  unavailable for your interpreter/platform, install the matching builds for your
  platform instead — the pipeline does not depend on the exact patch version.
- The server reads `webxr/` fresh per request, so frontend edits show on refresh
  with no restart.

---

## 5. Docker

The image installs `git-lfs`, resolves LFS blobs at build, and pre-downloads CLIP so
the first request is instant and runtime needs no Hugging Face access.

```bash
docker build -t timeline-vlm-xr .
docker run -p 7860:7860 timeline-vlm-xr
# → http://localhost:7860
```

Notes for reviewers building the image:
- The build runs `git lfs pull || true`. If your build context already contains the
  real blobs (e.g. you ran `git lfs pull` before building), they are used directly;
  otherwise the build attempts to fetch them. **Confirm the blobs are present in the
  context** for a fully offline, reproducible build.
- A `HEALTHCHECK` probes `/` via `urllib` (the slim image has no `curl`).
- `python:3.13-slim` matches the pinned torch wheels.

### Compose / deployment (optional)
- `docker-compose.yml` runs the container behind a Traefik reverse proxy with
  automatic Let's Encrypt TLS (used for the public headset-accessible deployment).
  It expects an external `proxy` network and is **not** needed to reproduce results.
  and publishes port 7860 directly (no Traefik). `docker compose up` auto-merges it;
  to reproduce the production setup, run `docker compose -f docker-compose.yml up`.

---

## 6. Method / pipeline (what to reproduce)

All steps are deterministic; see `server.py` line references.

1. **Load** precomputed embeddings + year labels for the backbone
   (`_load_embeddings`, `server.py:86`). Years that don't parse as `int` default to
   `1900` (`build_kpca_and_curve`, `server.py:72`).
2. **Reduce** the embeddings to 3-D with
   `KernelPCA(n_components=3, kernel="cosine")` (`server.py:76`).
3. **Fit a Bézier time-axis:** sample `n_ctrl` evenly-spaced control points from the
   year-ordered reduced points and evaluate a 1000-point Bézier via De Casteljau
   (`bezier_curve` / `_de_casteljau`, `server.py:52–60`). Default `n_ctrl=300`.
4. **Project** every timeline point onto its nearest curve point
   (`project_onto_curve`, `server.py:63`) — this snaps the cloud onto the smooth axis.
5. **Normalize** the cloud + curve into a ~2 m cube (`CUBE_EDGE = 2.0`,
   `_fit_normalizer` / `_apply_norm`, `server.py:104–116`) so the scene is
   comfortable at room scale in VR. The fitted center+scale are stored per backbone
   so user predictions land in the same coordinate frame.
6. **Serve** the scene JSON, cached per `(backbone, n_ctrl)` in `_scene_cache`
   (`get_scene`, `server.py:131`).

**Prediction** (`/api/predict`, `server.py:194`):
CLIP-encode the uploaded image → transform by the **same fitted KPCA** →
`project_onto_curve` onto the same curve → take the timeline label nearest (by curve
index) to the projected point as the **predicted year** → return the marker position
in normalized scene coordinates. Inference is serialized under `_PREDICT_LOCK`
because the model/sklearn state is not thread-safe.

---

## 7. HTTP API surface

| Method & path | Purpose |
|---------------|---------|
| `GET /` | `webxr/index.html` (app.js URL is mtime-stamped to bust caches) |
| `GET /api/timeline?backbone=clip&n_ctrl=300` | Scene JSON: `points`, `rawPoints`, `curve`, `years`, `labels`, `yearRange` |
| `POST /api/predict` (multipart `file`, `backbone=clip`) | `{name, year, position}` |
| `GET /api/examples` | Bundled demo image filenames (in-headset shelf) |
| `GET /phone`, `POST /api/share`, `GET /api/share/stream` (SSE), `GET /api/share/img/{id}` | Phone → headset photo sharing (demo convenience; not part of the method) |

---

## 8. Verifying a successful reproduction

After `python server.py` (or the Docker container) is up:

```bash
# 1. Scene JSON loads and spans the full year range.
curl -s 'http://localhost:7860/api/timeline?backbone=clip' \
  | python -c "import sys,json; d=json.load(sys.stdin); print('yearRange', d['yearRange'], 'points', len(d['points']))"
# expected: yearRange [1700, 2024]  points 325

# 2. Image prediction returns a plausible year for a bundled example.
curl -s -F file=@examples/iPhone_2016.jpg -F backbone=clip \
  http://localhost:7860/api/predict
# expected: {"name":"iPhone_2016.jpg","year":"<year>","position":[x,y,z]}

# 3. Non-clip prediction is rejected (documented limitation).
curl -s -F file=@examples/iPhone_2016.jpg -F backbone=eva \
  http://localhost:7860/api/predict
# expected: HTTP 400, "Image prediction is only available with the 'clip' backbone."
```

Bundled examples (with their ground-truth years in the filename) for qualitative
checking: `Ferrari_1994.jpg`, `HMS_Hotspur_1870.jpg`, `Lotus_2022.jpg`,
`iPhone_2016.jpg`.

> Prediction reproduces the upstream pipeline; treat the per-image estimates as
> demonstrative, not as a benchmark. The contribution of *this* artifact is the
> immersive WebXR rendering and interaction, not new accuracy numbers.

---

## 9. Reproducing the VR experience

WebXR needs a secure context. To put the demo on a headset over your LAN:

1. Run the server (local or Docker).
2. Expose it over **HTTPS** — e.g. a reverse proxy with TLS (see
   `docker-compose.yml` + Traefik) or an HTTPS tunnel. `localhost` is exempt from
   the HTTPS requirement for same-device testing.
3. In the headset browser, open the HTTPS URL. The **"Enter VR"** button appears
   when an `immersive-vr` device is detected.

**Controls:** *Trigger* (either controller) grabs the model to move/rotate it;
*right thumbstick* (vertical) scales it. Desktop: drag to orbit, scroll to zoom.
