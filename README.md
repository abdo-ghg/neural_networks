# neural_networks — Emotion detection experiments

This repository hosts facial expression recognition (FER) experiments and model artifacts. The root README focuses on the **Emotion detection** experiments; the `tasks` folder is intentionally ignored here.

## Overview
- Purpose: research and experiments for FER using CNNs and lightweight transformers.
- Main approaches: ResNet-style models, VGG variants, MobileViT+VGG hybrid, EfficientNetV2, ConvNeXt, and ViT/DeiT experiments.

## Key Notebooks (Emotion detection)
- `ResNet.ipynb` — grayscale input, full-data training, L2 + dropout regularization, cosine LR schedule, TTA and submission generation.
- `mobilevit_vgg_hybrid.ipynb` — custom VGG backbone fused with MobileViT-Small (via `timm`), channel adapter for grayscale→RGB, differential LR, TTA, saves `hybrid_submission.csv`.
- `EfficientNetV2-S.ipynb`, `DeiTScratch.ipynb`, `convnexttiny/convnexttiny-scratch.ipynb`, `vgg/vgg-updated.ipynb`, and experiments in `vit/` — additional backbone experiments.

## Artifacts
See `Emotion detection/files/` for model checkpoints and submission CSVs (examples: `best_vgg0.pth`, `pytorch_model.pt`, `submission.csv`).

## Running the Notebooks Locally
1. Prepare a Python environment with required libraries (PyTorch/tensorflow, timm, scikit-learn, pandas, numpy, matplotlib, seaborn). Example install commands:

```bash
python -m pip install torch torchvision torchaudio
python -m pip install tensorflow   # only if running TF notebooks
python -m pip install timm pandas numpy matplotlib seaborn scikit-learn
```

2. Download or place the FER dataset locally. Notebooks expect a directory structure with class subfolders under a training folder (update `DATA_DIR` / `TEST_DIR` variables inside notebooks to point to your local paths).

3. Open the desired `.ipynb` in VS Code or Jupyter and run cells in order. Save any produced artifacts (e.g., `best_model.keras`, `best_hybrid.pth`).

## Notes & Recommendations
- Many notebooks use grayscale face crops (single channel). Several models adapt pretrained RGB backbones by adding a channel adapter (1→3).
- Notebooks often reference Kaggle paths — update these paths before running locally.
- If you want, I can:
  - produce a consolidated `requirements.txt` for the Emotion detection environment,
  - add per-notebook README snippets with expected `DATA_DIR` values,
  - or make the notebooks accept a `DATA_DIR` environment variable.

---
File updated to focus on Emotion detection; `tasks` content intentionally excluded.
