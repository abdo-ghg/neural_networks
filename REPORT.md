# Project Report — neural_networks

This document summarizes the contents of the repository, explains the main experiments, notebooks, trained artifacts, and how to run the included tasks and apps.

**Repository Summary**
- **Project**: neural_networks — collection of experiments for facial expression recognition and small coursework tasks.
- **Top-level folders**: [Emotion detection](Emotion%20detection) (notebooks, models, submissions), [tasks](tasks) (Coursework: Task1, Task 2).

**Emotion detection** (folder)
- **Purpose**: experiments with several CNN / transformer-based architectures for facial expression recognition (FER). Notebooks contain data loading, model definitions, training loops, TTA inference and submission generation.
- **Key notebooks**:
  - [ResNet.ipynb](Emotion%20detection/ResNet.ipynb)
    - Grayscale input (single channel), IMG_SIZE=96, NUM_CLASSES=6.
    - Loads all training images, computes class weights, applies data augmentation (flip/brightness/contrast/noise), builds a ResNet-style model with L2 regularization + dropout, uses cosine-decayed LR, saves best model to `best_model.keras` and produces `submission.csv`.
  - [mobilevit_vgg_hybrid.ipynb](Emotion%20detection/mobilevit_vgg_hybrid.ipynb)
    - Hybrid architecture: custom VGG backbone (local features) + MobileViT-Small (global features via `timm`) fused with 1×1 conv.
    - Handles grayscale input via a channel adapter (1→3), uses pretrained MobileViT, differential learning rates (small for pretrained backbone, larger for newly initialized layers), uses TTA and saves `hybrid_submission.csv`.
  - [EfficientNetV2-S.ipynb](Emotion%20detection/EfficientNetV2-S.ipynb)
    - (Notebook present; likely explores EfficientNetV2-S backbone for FER.)
  - [DeiTScratch.ipynb](Emotion%20detection/DeiTScratch.ipynb)
    - (Notebook present; experiments with DeiT-style transformer trained from scratch.)
  - [convnexttiny/convnexttiny-scratch.ipynb](Emotion%20detection/convnexttiny/convnexttiny-scratch.ipynb)
    - (ConvNeXt Tiny experiments.)
  - [vgg/vgg-updated.ipynb](Emotion%20detection/vgg/vgg-updated.ipynb)
    - VGG-based model variants, training and inference cells present.
  - [vit/](Emotion%20detection/vit) (folder)
    - Additional transformer-based experiments.
- **Artifacts and submissions**: see [Emotion detection/files](Emotion%20detection/files)
  - Example trained weights: `best_vgg0.pth`, `pytorch_model.pt`.
  - Submission CSVs: `submission.csv`, `submission_vgg.csv`, `submission1.csv`, `submission (29).csv`.

Notes on datasets: Many notebooks reference datasets on Kaggle (absolute paths like `/kaggle/input/...`). When running locally, point the notebook variables to a local dataset folder (for example, download the competition dataset into a local folder and update `DATA_DIR` / `TEST_DIR`).

**tasks** (folder)
- **Purpose**: coursework-style tasks implementing small models and interactive apps using the penguins dataset.
- **Task1** (perceptron & Adaline)
  - Files: [tasks/Task1/app.py](tasks/Task1/app.py), [tasks/Task1/SLP_ADALINE.py](tasks/Task1/SLP_ADALINE.py), [tasks/Task1/penguins.csv](tasks/Task1/penguins.csv).
  - Functionality: Streamlit app to train and visualize a single-layer perceptron (SLP) and Adaline on the penguins dataset. Includes preprocessing (imputation, scaling, one-hot encoding), train/test split, training loops, metrics, decision-boundary plotting and confusion matrix.
  - Run: from repository root run the Streamlit app:

```bash
streamlit run "tasks/Task1/app.py"
```

- **Task 2** (micrograd MLP & Streamlit)
  - Files: [tasks/Task 2/main.py](tasks/Task%202/main.py), [tasks/Task 2/model.py](tasks/Task%202/model.py), [tasks/Task 2/micrograd/](tasks/Task%202/micrograd), [tasks/Task 2/streamlit_app.py](tasks/Task%202/streamlit_app.py), [tasks/Task 2/penguins.csv](tasks/Task%202/penguins.csv), [tasks/Task 2/pyproject.toml](tasks/Task%202/pyproject.toml).
  - Functionality: lightweight autograd (`micrograd`) based MLP implementation, training loop and a Streamlit dashboard that lets you configure network hyperparameters, run training, view loss/accuracy, confusion matrix and make single-sample predictions.
  - Run: install dependencies listed in `pyproject.toml` then run:

```bash
python -m pip install -r tasks/Task%202/requirements.txt  # or install from pyproject dependencies
streamlit run "tasks/Task 2/streamlit_app.py"
```

  - Notes: `pyproject.toml` in Task 2 lists core dependencies (numpy, pandas, scikit-learn, streamlit, seaborn, matplotlib). It specifies requires-python >=3.13 — if that is too strict for your environment, install the listed libraries on your Python version (3.8+ typically works for these libs).

**How to run the notebooks and training experiments**
- Notebook environment: open the `.ipynb` file in VS Code / Jupyter and select a Python kernel with required packages (PyTorch, TensorFlow, timm, torchvision, timm, scikit-learn, pandas, numpy, matplotlib, seaborn). Example install commands:

```bash
# PyTorch (choose appropriate CUDA/cuDNN version) e.g. CPU-only
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# TensorFlow (if running TF notebooks)
python -m pip install tensorflow

# Other common packages
python -m pip install timm pandas numpy matplotlib seaborn scikit-learn streamlit
```

- Running a training notebook:
  - Ensure dataset paths in the notebook point to local folders with the same structure used in the code (i.e., class subfolders for training images).
  - Run cells in order; save model artifacts that appear in the notebook (e.g., `best_model.keras`, `best_hybrid.pth`).

**Notable implementation details & conventions**
- Many notebooks use grayscale face crops (single channel) and normalize images to [0,1] or standardized tensors.
- `mobilevit_vgg_hybrid.ipynb` includes Arabic comments explaining the architecture and design choices.
- Training loops often use TTA (Test Time Augmentation) for inference averaging and class-weighting for imbalance.

**Files of interest**
- [Emotion detection/files/best_vgg0.pth](Emotion%20detection/files/best_vgg0.pth) — pre-trained PyTorch checkpoint (example).
- [Emotion detection/files/pytorch_model.pt](Emotion%20detection/files/pytorch_model.pt) — another saved model artifact.
- [tasks/Task 2/pyproject.toml](tasks/Task%202/pyproject.toml) — dependency list for Task 2.

**Recommended next steps**
- Add a top-level `requirements.txt` that aggregates all dependencies used across notebooks and tasks.
- Add small README files inside each subfolder describing dataset paths required and any external downloads (ImageNet pretrained weights are pulled via `timm`).
- Add consistent dataset loading helpers so notebooks can be run locally without editing many cells.

If you want, I can:
- generate a consolidated `requirements.txt` for the project,
- update notebooks to load datasets from a configurable `DATA_DIR` environment variable,
- or produce brief per-notebook README summaries with expected paths and sample commands to reproduce results.

---
Generated on: 2026-05-22
