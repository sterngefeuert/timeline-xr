# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An immersive **WebXR** demo for the paper *Can Vision-Language Models Tell When Things Were
Made?* (arxiv 2510.19559). It renders the temporal manifold (CLIP timeline embeddings,
1700–2024) as a 3-D point cloud you can explore in VR or orbit on desktop, and estimates the
year of an uploaded image by projecting its CLIP embedding onto a learned Bézier time-axis.

The backend is a single standalone FastAPI server, `server.py` (~240 lines). The frontend is
plain Three.js in `webxr/` (`index.html` + `app.js`, no build step). There are no tests and no
linter config.

This project was split out of the Gradio demo (`github.com/TekayaNidham/timeline-vlm`);
`server.py` deliberately does NOT import that app — it re-derives the ~30 lines of pipeline math.

## Commands

```bash
git lfs pull                                  # REQUIRED before first run — see LFS gotcha
pip install -r requirements.txt
python server.py                              # FastAPI + WebXR on 0.0.0.0:7860 (PORT env overrides)

docker build -t timeline-vlm-xr . && docker run -p 7860:7860 timeline-vlm-xr
```

## Critical environment gotchas

- **Embeddings + examples are Git LFS.** `encodings/*.npy` and `examples/*.jpg` are tracked via
  `.gitattributes`. A fresh clone has small pointer files; `np.load()` then fails with a clear
  "is a Git LFS pointer" error (see `_load_embeddings`). Run `git lfs pull` first.
- **torch pins.** `requirements.txt` pins `torch==2.6.0` / `torchvision==0.21.0`. If those wheels
  aren't available for your interpreter, install matching builds instead of the pinned versions.
- **Prediction is clip-only.** CLIP ViT-B/32 is the only encoder loaded, so `/api/predict` rejects
  `backbone != "clip"`. The `eva` backbone is timeline-visualization only.

## Architecture / pipeline

1. `encodings/` holds precomputed timeline embeddings (`.npy`) + a labels file (`.txt`, one year
   per line). `DATA_PATHS` maps a backbone (`clip`, `eva`) to its `{embeddings, labels}` pair.
2. `build_kpca_and_curve` reduces embeddings with `KernelPCA(n_components=3, kernel="cosine")` and
   fits a Bézier curve (`bezier_curve` / `_de_casteljau`) through `n_ctrl` evenly-sampled control
   points — a smooth 1-D temporal axis.
3. `get_scene` normalizes the reduced cloud + curve into a ~2 m cube (`_fit_normalizer` /
   `_apply_norm`) and caches the JSON per `(backbone, n_ctrl)` in `_scene_cache`.
4. `GET /api/timeline` serves that scene JSON; `POST /api/predict` CLIP-encodes an image,
   transforms it by the *same fitted KPCA*, `project_onto_curve` snaps it to the nearest curve
   point, and the nearest timeline label there is the predicted year.

## HTTP surface (server.py)

- `GET /` → `webxr/index.html`; `/` and `/examples` are also mounted as static dirs.
- `GET /api/timeline?backbone=clip&n_ctrl=300` → scene JSON (points, rawPoints, curve, years).
- `POST /api/predict` (multipart `file`, `backbone=clip`) → `{name, year, position}`. Inference is
  serialized under `_PREDICT_LOCK` (model/sklearn state isn't thread-safe); the endpoint is a sync
  `def` so Starlette runs it off the event loop.
- `GET /api/examples` → bundled demo image filenames (the in-headset shelf).

## Known quirks (verify before "fixing")

- Years that don't parse as `int` default to `1900` (`build_kpca_and_curve`).
- The pipeline math is duplicated from the upstream Gradio app on purpose — keep in sync if it changes.
