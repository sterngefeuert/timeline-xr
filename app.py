import asyncio
import sys

# Python 3.13 changed asyncio GC behavior - initialize a fresh event loop early
# to prevent BaseEventLoop.__del__ from corrupting the loop used by Gradio/uvicorn.
if sys.version_info >= (3, 13):
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

import os
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from sklearn.decomposition import KernelPCA
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import gradio as gr

print("=== app.py starting ===", flush=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading CLIP on {device}...", flush=True)
_hf_model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(_hf_model_id)
clip_model = CLIPModel.from_pretrained(_hf_model_id).to(device)
clip_model.eval()
print("=== CLIP model loaded ===", flush=True)

# Example images
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
EXAMPLE_IMAGES = [
    ("HMS Hotspur",  "~1870", "HMS_Hotspur_1870.jpg"),
    ("Ferrari F355", "~1994", "Ferrari_1994.jpg"),
    ("iPhone 7",     "~2016", "iPhone_2016.jpg"),
    ("Lotus Eletre", "~2022", "Lotus_2022.jpg"),
]
# Pre-load as PIL so the gallery uses base64 serialisation (no file-server needed)
try:
    _EXAMPLE_PILS = [
        (Image.open(os.path.join(EXAMPLES_DIR, fn)).copy(), f"{nm} ({yr})")
        for nm, yr, fn in EXAMPLE_IMAGES
    ]
    print(f"=== Loaded {len(_EXAMPLE_PILS)} example images ===", flush=True)
except Exception as e:
    print(f"Warning: example images not loaded: {e}", flush=True)
    _EXAMPLE_PILS = []

# Data
DATA_PATHS = {
    "clip": {"embeddings": "encodings/timeline_embeddings.npy",
             "labels":     "encodings/timeline_labels.txt"},
    "eva":  {"embeddings": "encodings/eva/eva_timeline_embeddings.npy",
             "labels":     "encodings/eva/eva_labels_timeline.txt"},
}
_cache = {}
print("=== Timeline data paths configured ===", flush=True)

def load_timeline(backbone):
    if backbone not in _cache:
        embs = np.load(DATA_PATHS[backbone]["embeddings"])
        with open(DATA_PATHS[backbone]["labels"]) as f:
            lbls = [l.strip() for l in f]
        _cache[backbone] = (embs, lbls)
    return _cache[backbone]

# Core math
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

# Distinct per-image colors for the timeline markers
_USER_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#9b5de5",
                "#f4a261", "#4cc9f0", "#fb5607", "#3a86ff"]

def _to_pil(img_or_path):
    """Accept PIL image, file path string, or numpy array."""
    if isinstance(img_or_path, Image.Image):
        return img_or_path.convert("RGB")
    return Image.open(img_or_path).convert("RGB")

# Figure builder
def _encode_user_images(user_images, kpca, curve, t_idx):
    """Shared: encode images, project onto curve, return (names, p_user, u_idx, pred_years)."""
    pil_imgs = [_to_pil(p) for p, _ in user_images]
    names    = [n for _, n in user_images]
    inputs   = clip_processor(images=pil_imgs, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        user_embs = clip_model.get_image_features(pixel_values=pixel_values).cpu().numpy()
    r_user        = kpca.transform(user_embs)
    p_user, u_idx = project_onto_curve(r_user, curve)
    pred_years = []
    for i in range(len(names)):
        near = int(np.argmin(np.abs(t_idx - u_idx[i])))
        pred_years.append(t_lbls_global[near] if hasattr(build_figure, '_t_lbls') else "?")
    return names, p_user, u_idx, pred_years

def build_figure_plotly(backbone, n_ctrl, show_original, user_images=None):
    t_embs, t_lbls = load_timeline(backbone)
    kpca, reduced, curve, proj_t, t_idx, years = build_kpca_and_curve(t_embs, t_lbls, n_ctrl)
    min_yr, max_yr = min(years), max(years)
    traces = []

    if show_original:
        traces.append(go.Scatter3d(
            x=reduced[:,0], y=reduced[:,1], z=reduced[:,2],
            mode="markers",
            marker=dict(size=3, color=years, colorscale="Viridis",
                        opacity=0.35, showscale=False),
            name="Original embeddings",
            hovertemplate="Year: %{text}<extra></extra>",
            text=t_lbls,
        ))

    traces.append(go.Scatter3d(
        x=proj_t[:,0], y=proj_t[:,1], z=proj_t[:,2],
        mode="markers",
        marker=dict(size=4, color=years, colorscale="Viridis",
                    colorbar=dict(title="Year", thickness=15, len=0.6),
                    cmin=min_yr, cmax=max_yr),
        name="Timeline",
        hovertemplate="Year: %{text}<extra></extra>",
        text=t_lbls,
    ))

    traces.append(go.Scatter3d(
        x=curve[:,0], y=curve[:,1], z=curve[:,2],
        mode="lines",
        line=dict(color="red", width=3),
        name="Bezier curve",
        hoverinfo="skip",
    ))

    step = max(1, (max_yr - min_yr) // 11)
    ann_years = set(range(min_yr, max_yr + 1, step))
    ax, ay, az, at = [], [], [], []
    for yr, pt in zip(years, proj_t):
        if yr in ann_years:
            ax.append(pt[0]); ay.append(pt[1]); az.append(pt[2]); at.append(str(yr))
    traces.append(go.Scatter3d(
        x=ax, y=ay, z=az, mode="text", text=at,
        textfont=dict(size=10, color="black"),
        showlegend=False, hoverinfo="skip",
    ))

    table_rows = []
    if user_images:
        pil_imgs = [_to_pil(p) for p, _ in user_images]
        names    = [n for _, n in user_images]
        inputs   = clip_processor(images=pil_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            user_embs = clip_model.get_image_features(pixel_values=pixel_values).cpu().numpy()
        r_user        = kpca.transform(user_embs)
        p_user, u_idx = project_onto_curve(r_user, curve)

        for i, name in enumerate(names):
            color   = _USER_COLORS[i % len(_USER_COLORS)]
            near    = int(np.argmin(np.abs(t_idx - u_idx[i])))
            pred_yr = t_lbls[near]
            pt      = p_user[i].copy(); pt[2] += 0.02

            traces.append(go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]],
                mode="markers+text",
                marker=dict(size=12, color=color, symbol="diamond",
                            line=dict(color="black", width=1.5)),
                text=[f"  {name}  (~{pred_yr})"],
                textfont=dict(size=10, color=color),
                textposition="middle right",
                name=name,
                hovertemplate=f"<b>{name}</b><br>~{pred_yr}<extra></extra>",
            ))
            table_rows.append((name, pred_yr))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.01, y=0.99),
        scene=dict(
            xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
        ),
        paper_bgcolor="white",
    )
    return fig, table_rows


def build_figure_matplotlib(backbone, n_ctrl, show_original, user_images=None):
    t_embs, t_lbls = load_timeline(backbone)
    kpca, reduced, curve, proj_t, t_idx, years = build_kpca_and_curve(t_embs, t_lbls, n_ctrl)
    min_yr, max_yr = min(years), max(years)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if show_original:
        ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2],
                   c=years, cmap="viridis", s=8, alpha=0.25, zorder=1)

    sc = ax.scatter(proj_t[:,0], proj_t[:,1], proj_t[:,2],
                    c=years, cmap="viridis", s=18, alpha=0.8, zorder=2)
    plt.colorbar(sc, ax=ax, label="Year", shrink=0.5, pad=0.1)

    ax.plot(curve[:,0], curve[:,1], curve[:,2],
            color="red", linewidth=2, zorder=3, label="Bezier curve")

    step = max(1, (max_yr - min_yr) // 11)
    ann_years = set(range(min_yr, max_yr + 1, step))
    for yr, pt in zip(years, proj_t):
        if yr in ann_years:
            ax.text(pt[0], pt[1], pt[2], str(yr), fontsize=7, color="black", zorder=4)

    table_rows = []
    if user_images:
        pil_imgs = [_to_pil(p) for p, _ in user_images]
        names    = [n for _, n in user_images]
        inputs   = clip_processor(images=pil_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            user_embs = clip_model.get_image_features(pixel_values=pixel_values).cpu().numpy()
        r_user        = kpca.transform(user_embs)
        p_user, u_idx = project_onto_curve(r_user, curve)

        for i, name in enumerate(names):
            color   = _USER_COLORS[i % len(_USER_COLORS)]
            near    = int(np.argmin(np.abs(t_idx - u_idx[i])))
            pred_yr = t_lbls[near]
            pt      = p_user[i].copy(); pt[2] += 0.02

            ax.scatter([pt[0]], [pt[1]], [pt[2]],
                       color=color, s=120, marker="D", edgecolors="black",
                       linewidths=1.2, zorder=5)
            ax.text(pt[0], pt[1], pt[2], f"  {name} (~{pred_yr})",
                    fontsize=8, color=color, zorder=6)
            table_rows.append((name, pred_yr))

    ax.set_xlabel("Dim 1", fontsize=9)
    ax.set_ylabel("Dim 2", fontsize=9)
    ax.set_zlabel("Dim 3", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    return fig, table_rows


def build_figure(backbone, n_ctrl, show_original, user_images=None, viz="Plotly"):
    if viz == "Matplotlib":
        fig, table_rows = build_figure_matplotlib(backbone, n_ctrl, show_original, user_images)
    else:
        fig, table_rows = build_figure_plotly(backbone, n_ctrl, show_original, user_images)

    table = ""
    if table_rows:
        header = "### Predictions\n| Image | Estimated year |\n|---|---|\n"
        rows   = "\n".join(f"| {n} | **{y}** |" for n, y in table_rows)
        table  = header + rows
    return fig, table

# Callbacks
def show_timeline(backbone, n_ctrl, show_original, viz):
    fig, _ = build_figure(backbone, n_ctrl, show_original, viz=viz)
    return fig, ""

def on_example_select(evt: gr.SelectData, state, backbone, n_ctrl, show_original, viz):
    name, _, filename = EXAMPLE_IMAGES[evt.index]
    path = os.path.join(EXAMPLES_DIR, filename)
    if not any(n == name for _, n in state):
        # Store PIL image directly alongside the name to avoid file-path serving issues
        pil_img = Image.open(path).convert("RGB")
        state = state + [(pil_img, name)]
    fig, table = build_figure(backbone, n_ctrl, show_original, state, viz=viz)
    return state, fig, table

def on_upload(files, state, backbone, n_ctrl, show_original, viz):
    if not files:
        fig, _ = build_figure(backbone, n_ctrl, show_original, viz=viz)
        return state, fig, ""
    existing = {n for _, n in state}
    for f in files:
        path  = f.name if hasattr(f, "name") else str(f)
        fname = os.path.basename(path)
        if fname not in existing:
            state = state + [(path, fname)]
            existing.add(fname)
    fig, table = build_figure(backbone, n_ctrl, show_original, state, viz=viz)
    return state, fig, table

def refresh_plot(backbone, n_ctrl, show_original, viz, state):
    fig, table = build_figure(backbone, n_ctrl, show_original, state if state else None, viz=viz)
    return fig, table

# UI
APP_CSS = """
:root {
  --tl-bg: #f6f8fb;
  --tl-ink: #18202f;
  --tl-muted: #617086;
  --tl-line: #d8e0ec;
  --tl-panel: #ffffff;
  --tl-panel-soft: #eef4ff;
  --tl-accent: #315fbe;
  --tl-accent-dark: #203f80;
}
.gradio-container {
  max-width: 1560px !important;
  margin: 0 auto !important;
  background: var(--tl-bg) !important;
  color: var(--tl-ink) !important;
}
.tl-hero {
  border: 1px solid var(--tl-line);
  border-radius: 18px;
  padding: 22px 26px 20px;
  margin-bottom: 16px;
  background:
    linear-gradient(135deg, rgba(49,95,190,.10), rgba(10,15,25,0) 44%),
    var(--tl-panel);
  box-shadow: 0 18px 50px rgba(29, 43, 72, .08);
}
.tl-title {
  margin: 0 0 8px;
  font-size: 34px;
  line-height: 1.05;
  letter-spacing: 0;
  color: var(--tl-ink);
}
.tl-lede {
  max-width: 920px;
  margin: 0;
  color: var(--tl-muted);
  font-size: 15px;
  line-height: 1.55;
}
.tl-links {
  margin-top: 12px;
  display: flex;
  gap: 16px;
  font-size: 13px;
  font-weight: 650;
}
.tl-links a { color: var(--tl-accent-dark); text-decoration: none; }
.tl-links a:hover { text-decoration: underline; }
.tl-panel, .tl-main, .tl-examples {
  border: 1px solid var(--tl-line);
  border-radius: 16px;
  padding: 14px;
  background: rgba(255,255,255,.92);
  box-shadow: 0 10px 30px rgba(29,43,72,.06);
}
.tl-panel h3, .tl-examples h3 {
  margin-top: 0 !important;
  color: var(--tl-ink);
  font-size: 14px;
  letter-spacing: .02em;
  text-transform: uppercase;
}
.tl-note {
  border-top: 1px solid var(--tl-line);
  margin-top: 16px;
  padding: 14px 4px 2px;
  color: var(--tl-muted);
  font-size: 13px;
  line-height: 1.55;
}
.tl-note strong { color: var(--tl-ink); }
.tl-workspace {
  align-items: stretch !important;
  min-height: calc(100vh - 210px);
}
.tl-workspace > .tl-panel,
.tl-workspace > .tl-main,
.tl-workspace > .tl-examples {
  min-height: calc(100vh - 210px);
}
.tl-examples {
  display: flex !important;
  flex-direction: column !important;
  flex-wrap: nowrap !important;
  overflow: hidden !important;
}
.tl-examples > .block {
  width: 100% !important;
  max-width: 100% !important;
}
.tl-example-gallery {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  height: auto !important;
  width: 100% !important;
  max-width: 100% !important;
  overflow: hidden !important;
}
.tl-example-gallery > div,
.tl-example-gallery .gallery-container,
.tl-example-gallery .grid-wrap {
  width: 100% !important;
  max-width: 100% !important;
}
.tl-example-gallery .gallery-container,
.tl-example-gallery .grid-wrap {
  height: 100% !important;
  overflow: hidden auto !important;
}
.tl-example-gallery img {
  max-width: 100% !important;
  object-fit: cover !important;
}
button.primary, button.secondary {
  border-radius: 10px !important;
  font-weight: 650 !important;
}
"""

DESCRIPTION = """
<div class="tl-hero">
  <h1 class="tl-title">Temporal Manifold Explorer</h1>
  <p class="tl-lede">Place visual artifacts on a CLIP-derived timeline and inspect how the model organizes first-appearance years inside its embedding space.</p>
  <div class="tl-links">
    <a href="https://arxiv.org/pdf/2510.19559" target="_blank">Companion paper</a>
    <a href="https://github.com/TekayaNidham/timeline-vlm" target="_blank">Source repository</a>
  </div>
</div>
"""

HOW_IT_WORKS = """
<div class="tl-note">
  <strong>Pipeline.</strong> Year prompts are encoded with CLIP, reduced with cosine-kernel KPCA, and threaded by a Bezier time-axis. Uploaded images are projected into the same space and snapped to the nearest point on the curve to estimate their year.
</div>
"""

with gr.Blocks(title="Temporal Manifold Explorer") as demo:
    gr.Markdown(DESCRIPTION)
    img_state = gr.State([])   # list of (path: str, name: str)

    with gr.Row(elem_classes=["tl-workspace"]):
        # Left: controls + upload
        with gr.Column(scale=1, min_width=250, elem_classes=["tl-panel"]):
            backbone_input = gr.Radio(
                choices=[("CLIP ViT-B/32", "clip"), ("EVA01-CLIP-g-14", "eva")],
                value="clip", label="Embedding backbone"
            )
            viz_input = gr.Radio(
                choices=["Plotly", "Matplotlib"],
                value="Plotly", label="Visualization"
            )
            with gr.Accordion("Advanced options", open=False):
                ctrl_input = gr.Slider(10, 1000, value=300, step=10,
                                       label="Bezier control points")
                orig_input = gr.Checkbox(value=False,
                                         label="Show raw (pre-projection) embeddings")
            timeline_btn = gr.Button("Recompute timeline", variant="secondary")

            gr.Markdown("### Query images")
            files_input = gr.File(
                label="Upload artifact images",
                file_count="multiple",
                file_types=["image"],
            )

        # Center: plot + predictions
        with gr.Column(scale=3, elem_classes=["tl-main"]):
            plot_output = gr.Plot(label="Temporal manifold")
            pred_output = gr.Markdown()

        # Right: example images
        with gr.Column(scale=1, min_width=230, elem_classes=["tl-examples"]):
            gr.Markdown("### Artifact set\n*Select an example to place it on the timeline.*")
            example_gallery = gr.Gallery(
                value=_EXAMPLE_PILS if _EXAMPLE_PILS else None,
                show_label=False,
                columns=2,
                rows=2,
                height=620,
                elem_classes=["tl-example-gallery"],
                allow_preview=False,
                interactive=False,
            )

    gr.Markdown(HOW_IT_WORKS)

    # Events
    _common_inputs  = [backbone_input, ctrl_input, orig_input, viz_input]
    _common_outputs = [plot_output, pred_output]

    demo.load(fn=show_timeline,
              inputs=_common_inputs,
              outputs=_common_outputs)

    timeline_btn.click(fn=refresh_plot,
                       inputs=_common_inputs + [img_state],
                       outputs=_common_outputs)

    # Swap viz on radio change (re-render current state)
    viz_input.change(fn=refresh_plot,
                     inputs=_common_inputs + [img_state],
                     outputs=_common_outputs)

    example_gallery.select(fn=on_example_select,
                            inputs=[img_state] + _common_inputs,
                            outputs=[img_state] + _common_outputs)

    files_input.upload(fn=on_upload,
                       inputs=[files_input, img_state] + _common_inputs,
                       outputs=[img_state] + _common_outputs)

print("=== Gradio UI built, launching... ===", flush=True)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        css=APP_CSS,
        allowed_paths=[EXAMPLES_DIR],
        share=os.environ.get("GRADIO_SHARE") == "1",
    )
