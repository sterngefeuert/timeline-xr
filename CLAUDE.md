# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A demo for the paper *Can Vision-Language Models Tell When Things Were Made?* (arxiv 2510.19559).
Upload an image of a visual artifact; the model estimates the year it was created by
projecting its CLIP embedding onto a learned temporal manifold. The entire app is `app.py`
(~440 lines, single file). There are no tests, no linter config, and no build step.

## Commands

```bash
python app.py            # launches the Gradio UI on 0.0.0.0 (default port 7860)
git lfs pull             # REQUIRED before first run — see "Embeddings are LFS" below
git push origin main     # this IS the deploy: origin is the HF Space
```

## Critical environment gotchas

- **Embeddings are Git LFS.** `encodings/*.npy` are tracked via `.gitattributes` (`*.npy filter=lfs`).
  A fresh clone has ~131-byte pointer files, and `np.load()` will fail. Run `git lfs pull`
  (install git-lfs first — it is not present in this environment).
- **`requirements.txt` is HF-specific.** Lines 1–2 pin `torch`/`torchvision` to
  `cp313 ... linux_x86_64` CPU wheels because the HF Space runs Python 3.13. These will not
  install on macOS or other Python versions. For local work, install torch/torchvision
  appropriate to the local interpreter instead of using the pinned URLs verbatim.
- **The Python 3.13 asyncio shim** at the top of `app.py` (creating a fresh event loop) exists
  to stop `BaseEventLoop.__del__` from corrupting the Gradio/uvicorn loop. Don't remove it.
- **Deployment = pushing to the HF Space.** `origin` is
  `huggingface.co/spaces/Nidhamtek/timeline-vlm`. The YAML frontmatter in `README.md`
  (`sdk`, `sdk_version: 6.10.0`, `app_file: app.py`) is the Space's build config — keep
  `sdk_version` consistent with the `gradio` behavior the code relies on. The git history is
  almost entirely HF-deployment fixes (torch wheels, Python version, health check).

## Architecture / pipeline

The core idea (see `build_kpca_and_curve` and the figure builders):

1. `encodings/` holds **precomputed timeline embeddings** (`.npy`) plus a labels file
   (`.txt`, one year per line, ascending from 1700). `DATA_PATHS` maps a backbone name to
   its `{embeddings, labels}` pair; `load_timeline()` caches per backbone in `_cache`.
2. `KernelPCA(n_components=3, kernel="cosine")` reduces timeline embeddings to 3-D — the
   "temporal manifold".
3. A **Bézier curve** (`bezier_curve` / `_de_casteljau`) is fitted through `n_ctrl` control
   points sampled evenly along the reduced timeline, giving a smooth 1-D temporal axis.
4. A user image is CLIP-encoded, transformed by the *same fitted KPCA*, then
   `project_onto_curve` snaps it to the nearest curve point. The nearest timeline label at
   that curve index is the predicted year.

Rendering: `build_figure` dispatches to `build_figure_plotly` (interactive 3-D, default) or
`build_figure_matplotlib` (static 3-D). The user-image encode+project block is duplicated
inline in both builders.

UI state: a single `gr.State([])` holds a list of `(PIL image | path, name)` tuples.
Callbacks (`on_example_select`, `on_upload`, `clear_all`, `refresh_plot`) all rebuild the
whole figure from that list — there is no incremental update.

## Known quirks (verify before "fixing")

- **The backbone radio is not fully wired.** Selecting "eva" only swaps the *timeline*
  embeddings; user images are always encoded with the loaded `clip_model`
  (CLIP ViT-B/32, 512-d). KPCA is fit on the chosen backbone's timeline embeddings, so
  EVA + an uploaded image mixes mismatched embedding spaces. Treat EVA as timeline-only.
- **`_encode_user_images` is dead code.** It references `t_lbls_global` /
  `build_figure._t_lbls`, which are never defined. The live path is the inlined block in
  each `build_figure_*` function. Don't call or extend `_encode_user_images`; fix the
  inlined blocks instead.
- Years that don't parse as `int` default to `1900` (`build_kpca_and_curve`).
