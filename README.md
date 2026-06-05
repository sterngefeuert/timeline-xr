# 🕰️ Timeline-VLM — WebXR

An immersive WebXR view of the temporal manifold from
*[Can Vision-Language Models Tell When Things Were Made?](https://arxiv.org/pdf/2510.19559)*

CLIP timeline embeddings (1700–2024) are reduced to 3-D and rendered as a
year-coloured point cloud threaded by the learned Bézier time-axis. Walk through
it in VR, or orbit it on a desktop. Upload an image and its estimated position
drops into the scene as a marker.

📄 [Paper](https://arxiv.org/pdf/2510.19559) · 🧪 Gradio demo: [TekayaNidham/timeline-vlm](https://github.com/TekayaNidham/timeline-vlm)

## Run locally

```bash
git lfs pull                        # encodings/*.npy + examples/*.jpg are LFS-tracked
pip install -r requirements.txt     # see torch note below
python server.py                    # http://localhost:7860
```

Open `http://localhost:7860` — desktop works immediately (drag to orbit, scroll
to zoom, toggle raw embeddings, upload an image to estimate its year).

- **torch note:** `requirements.txt` pins `torch==2.6.0` / `torchvision==0.21.0`.
  If those wheels aren't available for your interpreter, install the
  `torch`/`torchvision` build that matches your platform instead.
- **VR headset:** WebXR requires HTTPS (or `localhost`). For a headset on your
  LAN, put it behind an HTTPS tunnel/reverse proxy; the "Enter VR" button appears
  when an `immersive-vr` device is available.

## VR controls

- **Trigger** (either controller): grab the model to move/rotate it.
- **Right thumbstick** (vertical): scale the model up/down.

## Docker

```bash
docker build -t timeline-vlm-xr .
docker run -p 7860:7860 timeline-vlm-xr
```

The image installs `git-lfs` and runs `git lfs pull` at build so the embedding
blobs resolve. To deploy as a Hugging Face **Docker** Space, create a Space with
`sdk: docker` and push this repo.

## How it works

`server.py` is a standalone FastAPI server (it does **not** import any Gradio
code). It:

1. loads precomputed timeline embeddings from `encodings/` (`clip` and `eva`
   backbones),
2. reduces them with `KernelPCA(n_components=3, kernel="cosine")` and fits a
   Bézier curve as a smooth 1-D time axis,
3. normalizes the scene into a ~2 m cube and serves it as JSON at `/api/timeline`,
4. on `POST /api/predict`, CLIP-encodes an uploaded image, projects it onto the
   curve, and returns the nearest year + its scene position.

The `webxr/` frontend (Three.js, loaded from CDN) renders that JSON and handles
WebXR. Image prediction is **clip-only** (CLIP ViT-B/32 is the only encoder
loaded); the `eva` backbone is timeline-visualization only.
