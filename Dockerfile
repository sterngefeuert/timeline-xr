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

EXPOSE 7860
ENV PORT=7860
CMD ["python", "server.py"]
