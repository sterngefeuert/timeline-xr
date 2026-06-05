# Timeline-VLM WebXR — immersive Three.js/WebXR view of the temporal manifold.
# Build & run locally (or deploy as a Hugging Face *Docker* Space):
#   docker build -t timeline-vlm-xr .
#   docker run -p 7860:7860 timeline-vlm-xr
#
# python:3.13 matches the pinned torch wheels in requirements.txt.
FROM python:3.13-slim

# git-lfs so the encodings/*.npy + examples/*.jpg LFS blobs resolve at build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Resolve LFS pointers if real blobs weren't included in the build context.
RUN git lfs pull || true

# Pre-download CLIP so the first request is instant and runtime needs no HF access.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    m='openai/clip-vit-base-patch32'; CLIPProcessor.from_pretrained(m); CLIPModel.from_pretrained(m)"

# slim image has no curl, so probe the lightest route ('/') with Python's urllib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

EXPOSE 7860
ENV PORT=7860
CMD ["python", "server.py"]
