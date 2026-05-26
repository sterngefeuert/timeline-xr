import asyncio
import sys

# Python 3.13 changed asyncio GC behaviour — initialise a fresh event loop early
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

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading CLIP on {device}…", flush=True)
_hf_model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(_hf_model_id)
clip_model = CLIPModel.from_pretrained(_hf_model_id).to(device)
clip_model.eval()
print("=== CLIP model loaded ===", flush=True)

# ── Example images ────────────────────────────────────────────────────────────
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

# ── Data ──────────────────────────────────────────────────────────────────────
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

# ── Core math ─────────────────────────────────────────────────────────────────
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

# ── Figure builder ────────────────────────────────────────────────────────────
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

# ── State helpers ─────────────────────────────────────────────────────────────
def _gallery_val(state):
    return [(p, n) for p, n in state] if state else []

# ── Callbacks ─────────────────────────────────────────────────────────────────
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
    return state, _gallery_val(state), fig, table

def on_upload(files, state, backbone, n_ctrl, show_original, viz):
    if not files:
        fig, _ = build_figure(backbone, n_ctrl, show_original, viz=viz)
        return state, _gallery_val(state), fig, ""
    existing = {n for _, n in state}
    for f in files:
        path  = f.name if hasattr(f, "name") else str(f)
        fname = os.path.basename(path)
        if fname not in existing:
            state = state + [(path, fname)]
            existing.add(fname)
    fig, table = build_figure(backbone, n_ctrl, show_original, state, viz=viz)
    return state, _gallery_val(state), fig, table

def clear_all(backbone, n_ctrl, show_original, viz):
    fig, _ = build_figure(backbone, n_ctrl, show_original, viz=viz)
    return [], [], fig, ""

def refresh_plot(backbone, n_ctrl, show_original, viz, state):
    fig, table = build_figure(backbone, n_ctrl, show_original, state if state else None, viz=viz)
    return fig, table

# ── UI ────────────────────────────────────────────────────────────────────────
DESCRIPTION = """
# Timeline-VLM

Upload images of visual artifacts and the model estimates **when they were created** by projecting their visual embeddings onto a learned temporal manifold.

📄 [Paper](https://arxiv.org/pdf/2510.19559) &nbsp;·&nbsp; 💻 [GitHub](https://github.com/TekayaNidham/timeline-vlm)
"""

HOW_IT_WORKS = """
---
### How it works
1. Text prompts (*"an artifact from the year XXXX"*) are encoded with CLIP to build a **temporal embedding space**
2. Kernel PCA (cosine kernel) reduces these to 3-D, revealing a *temporal manifold*
3. A **Bezier curve** is fitted through the manifold as a smooth temporal axis
4. Your image is encoded and **projected** onto that curve, and its position gives the estimated year
"""

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    img_state = gr.State([])   # list of (path: str, name: str)

    with gr.Row():
        # ── Left: controls + upload ───────────────────────────────────────────
        with gr.Column(scale=1, min_width=240):
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
            timeline_btn = gr.Button("🌐 Refresh timeline", variant="secondary")

            gr.Markdown("### 📂 Upload images")
            files_input = gr.File(
                label="Drop or click to upload",
                file_count="multiple",
                file_types=["image"],
            )

            gr.Markdown("**🗂️ On the timeline**")
            selected_gallery = gr.Gallery(
                show_label=False, columns=4, height=65,
                allow_preview=False, interactive=False,
            )
            clear_btn = gr.Button("🗑️ Clear all", variant="secondary", size="sm")

        # ── Centre: plot + predictions ────────────────────────────────────────
        with gr.Column(scale=3):
            plot_output = gr.Plot(label="Temporal embedding space")
            pred_output = gr.Markdown()

        # ── Right: example images ─────────────────────────────────────────────
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### 📸 Examples\n*Click to add to timeline*")
            example_gallery = gr.Gallery(
                value=_EXAMPLE_PILS if _EXAMPLE_PILS else None,
                show_label=False,
                columns=2,
                rows=2,
                height=250,
                allow_preview=False,
                interactive=False,
            )

    gr.Markdown(HOW_IT_WORKS)

    # ── Events ────────────────────────────────────────────────────────────────
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
                            outputs=[img_state, selected_gallery] + _common_outputs)

    files_input.upload(fn=on_upload,
                       inputs=[files_input, img_state] + _common_inputs,
                       outputs=[img_state, selected_gallery] + _common_outputs)

    clear_btn.click(fn=clear_all,
                    inputs=_common_inputs,
                    outputs=[img_state, selected_gallery] + _common_outputs)

print("=== Gradio UI built, launching... ===", flush=True)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        allowed_paths=[EXAMPLES_DIR],
        share=os.environ.get("GRADIO_SHARE") == "1",
    )
