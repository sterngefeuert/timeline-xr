# 🕰️ Timeline-VLM — WebXR

An immersive WebXR view of the temporal manifold from
*[Can Vision-Language Models Tell When Things Were Made?](https://arxiv.org/pdf/2510.19559)*

The CLIP timeline embeddings (1700–2024) are reduced to 3-D and rendered as a
year-coloured point cloud threaded by the learned Bézier time-axis. Walk through
it in VR, or orbit it on a desktop. Upload an image and its estimated position
drops into the scene as a marker.

This is **additive**: `app.py` (the live Gradio Hugging Face Space) is untouched.
The WebXR stack — `server.py` + `webxr/` — is a separate entry point.

## Run locally

```bash
git lfs pull                                              # encodings/*.npy are LFS pointers
pip install -r requirements.txt -r requirements-webxr.txt # see note below on torch
python server.py                                          # http://localhost:7860
```

Open `http://localhost:7860` — desktop works immediately (drag to orbit, scroll
to zoom, toggle raw embeddings, upload an image to estimate its year).

- **torch note:** lines 1–2 of `requirements.txt` pin Linux `cp313` CPU wheels
  (for the HF Space's Python 3.13). On macOS/other Python, install
  `torch`/`torchvision` for your interpreter instead of those URLs.
- **VR headset:** WebXR requires HTTPS (or `localhost`). For a headset on your
  LAN, put it behind an HTTPS tunnel/reverse proxy; the “Enter VR” button shows
  when an `immersive-vr` device is available.

## VR controls

- **Trigger** (either controller): grab the model to move/rotate it.
- **Right thumbstick** (vertical): scale the model up/down.

## Deploy (Hugging Face)

`origin` is the live **Gradio** Space (`sdk: gradio` in `README.md`) and still
runs `app.py`. To publish the WebXR version, do **not** edit `README.md` on
`main` — instead either:

1. **A separate Docker Space** (recommended, keeps both demos): create a new
   Space with `sdk: docker`, push this repo; the included `Dockerfile` runs
   `server.py`. Or
2. **Convert this Space:** change `README.md` frontmatter to `sdk: docker`
   (drop `sdk_version`/`app_file`) — this replaces the Gradio demo with WebXR.

Both need the `encodings/*.npy` LFS blobs (the `Dockerfile` runs `git lfs pull`).
