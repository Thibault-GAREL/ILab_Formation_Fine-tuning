# 🎨 Fine-tuning Stable Diffusion on My Chibi Drawings

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/Pytorch-2.5.1%2Bcu121-red.svg)
![Diffusers](https://img.shields.io/badge/diffusers-0.38-yellow.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <img src="data/1-raw/my_drawings/Capture Bienvenu.PNG" alt="Chibi Thibault waving hello" width="60%">
  <!-- <img src="data/1-raw/my_drawings/Capture bras croisés.PNG" alt="Chibi Thibault arms crossed" width="220">
  <img src="data/1-raw/my_drawings/Capture surprise.PNG" alt="Chibi Thibault surprised" width="220"> -->
</p>

---

## 📝 Project Description

This project fine-tunes **Stable Diffusion 1.5** on **15 hand-drawn chibi self-portraits** so the model learns to generate new images in my personal art style. It uses **LoRA (Low-Rank Adaptation)** via 🤗 `diffusers` + `peft` — a parameter-efficient method that produces a tiny ~40 MB adapter file instead of touching the base model weights.

The goal: a **simple, fast, reusable base project** for future style/character fine-tunings. The character trigger word is `thibchibi` — I use it in prompts to invoke the learned style.

---

## 🚨 Status — Work In Progress

This project is being built **step-by-step**. Current state:

  ✅ **Project structure** — `src/`, `data/`, `outputs/` skeleton ready

  ✅ **Config** ([src/config.py](src/config.py)) — LoRA hyperparameters tuned for **GTX 1660 Ti (6 GB VRAM)**

  ✅ **Data pipeline** ([src/data/make_dataset.py](src/data/make_dataset.py)) — auto captions from filenames, resize 512×512 with white padding

  ⏳ **Training script** — TODO ([src/models/train.py](src/models/train.py))

  ⏳ **Inference script** — TODO ([src/models/generate.py](src/models/generate.py))

  ⏳ **Gradio demo** — TODO ([src/demo.py](src/demo.py))

  ⏳ **Architecture diagram** — TODO (will be generated via the `canva-diagrams` skill)

### ⚠️ Pending install fix

`pip install diffusers gradio` failed mid-install due to a Windows file-lock on `_safetensors_rust.pyd` (another Python process held the file). The `safetensors` package was uninstalled but not reinstalled — **the venv is in a broken state until this is fixed**.

**To resolve** (close all VS Code Python interpreters / Jupyter kernels first):

```powershell
& c:\0-Code_py_temp\pytorch_cuda_env\Scripts\Activate.ps1
pip install --force-reinstall --no-deps safetensors
pip install diffusers gradio
python -c "import diffusers, gradio, safetensors; print(diffusers.__version__, gradio.__version__, safetensors.__version__)"
```

---

## ⚙️ Features

  🎨 **LoRA fine-tuning** of Stable Diffusion 1.5 on a tiny dataset (15 images)

  ⚡ **Trigger-word approach** — single token (`thibchibi`) shared across all training images, captions auto-generated from filenames

  🪶 **Lightweight output** — final adapter ~40 MB, base SD model stays untouched and reusable

  💾 **VRAM-friendly** — fp16, gradient checkpointing, gradient accumulation → fits on a 6 GB GPU

  🧪 **MLflow tracking** — loss curves, hyperparams, sample images logged per run

  🖼️ **Gradio demo** — local web UI to test prompts with the trained LoRA loaded

---

## ⚙️ How it works

  🖼️ **15 chibi drawings** are resized to 512×512 with white padding to preserve their aspect ratio.

  📝 **Captions are generated automatically** from filenames (e.g., `Capture Au revoir 3.PNG` → `"a thibchibi chibi character, waving goodbye"`).

  🧠 **LoRA adapters** are injected into the attention layers of the Stable Diffusion **UNet** (`peft` `LoraConfig` with rank=16).

  🎯 The base SD model is **frozen** — only the LoRA weights (~0.8% of total params) are trained.

  📉 Training runs for **~800 steps** with AdamW (lr=1e-4), batch size 1, gradient accumulation 4 → effective batch 4. Total time ≈ 20-30 min on GTX 1660 Ti.

  🔮 **At inference**, the LoRA is loaded on top of vanilla SD 1.5, and prompts containing `thibchibi` produce images in the learned style.

---

## 🗺️ Architecture Diagram

> 🚧 SVG diagram TODO — will be generated via the `canva-diagrams` skill once training is validated.

For now, the high-level flow:

```
                    ┌──────────────────────────────┐
                    │ 15 PNG drawings + captions   │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Stable Diffusion 1.5 (frozen)                              │
│                                                             │
│   Text Encoder ──► UNet ◄── LoRA adapters (TRAINED)         │
│        │             │                                      │
│        ▼             ▼                                      │
│      tokens      noise pred                                 │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │ outputs/models/lora_thibchibi│
                    │       (~40 MB adapter)       │
                    └──────────────────────────────┘
```

**Key hyperparameters** (see [src/config.py](src/config.py)):
- `BASE_MODEL = stable-diffusion-v1-5/stable-diffusion-v1-5`
- `TRIGGER_WORD = thibchibi`
- `LORA_RANK = 16`, `LORA_ALPHA = 16`
- `MAX_TRAIN_STEPS = 800`, `LEARNING_RATE = 1e-4`
- `MIXED_PRECISION = fp16`, `GRADIENT_CHECKPOINTING = True`

---

## 📂 Repository structure

```bash
├── data/
│   ├── 1-raw/
│   │   └── my_drawings/        # 15 chibi PNG (input)
│   ├── 2-processed/            # generated dataset (512x512 + metadata.jsonl)
│   └── 3-external/
│
├── outputs/
│   ├── models/                 # LoRA adapter weights end up here
│   ├── logs/                   # MLflow runs
│   └── results/                # generated samples
│
├── src/
│   ├── config.py               # pydantic settings (paths + hyperparams)
│   ├── utils.py
│   ├── data/
│   │   └── make_dataset.py     # PNG -> resize -> captions -> dataset HF
│   ├── models/
│   │   ├── model.py
│   │   └── train.py            # LoRA fine-tuning loop + MLflow
│   └── validation/
│       └── metrics.py
│
├── tests/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 💻 Run it on Your PC

Clone the repository:

```bash
git clone https://github.com/Thibault-GAREL/ILab_Formation_Fine_tuning.git
cd ILab_Formation_Fine_tuning
```

Activate the pre-configured PyTorch + CUDA env (Windows / PowerShell):

```powershell
& c:\0-Code_py_temp\pytorch_cuda_env\Scripts\Activate.ps1
```

Install the project-specific libraries (the rest is already in the venv):

```powershell
pip install diffusers gradio
```

⚠️ You need a **CUDA-compatible GPU** with at least **6 GB VRAM** to fine-tune locally. Otherwise, the same scripts can be executed on **RunPod** (rent an RTX 3090 / A4000 for ~0.25 $/h).

### 1. Prepare the dataset

```powershell
python -m src.data.make_dataset
```

Reads `data/1-raw/my_drawings/`, writes resized images + `metadata.jsonl` into `data/2-processed/thibchibi_dataset/`.

### 2. Train the LoRA

```powershell
python -m src.models.train
```

LoRA adapter saved to `outputs/models/lora_thibchibi/`. Training run logged to MLflow.

### 3. Generate images

```powershell
python -m src.models.generate --prompt "a thibchibi character riding a bike"
```

### 4. Launch the Gradio demo

```powershell
python -m src.demo
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## 📖 Inspiration / Sources

I used Claude Code to scaffold the LoRA fine-tuning pipeline (config, dataset prep, training loop) — the goal is to use this project as a **reusable base** for future style/character fine-tunings on small datasets.

Built on top of:
- 🤗 [diffusers](https://github.com/huggingface/diffusers) — Stable Diffusion implementation
- 🤗 [peft](https://github.com/huggingface/peft) — LoRA / parameter-efficient fine-tuning

Code created by me 😎, Thibault GAREL - [Github](https://github.com/Thibault-GAREL)
