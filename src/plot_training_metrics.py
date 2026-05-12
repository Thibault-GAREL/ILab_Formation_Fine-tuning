"""Génère 2 plots SVG (loss + LR schedule) à partir des metrics MLflow du dernier run.

Lit `outputs/logs/mlruns/<exp>/<run>/metrics/{loss,lr}` (format MLflow file-store)
et écrit `assets/loss_curve.svg` + `assets/lr_schedule.svg`.

Usage : python -m src.plot_training_metrics
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import config


ASSETS_DIR = config.OUTPUTS_LOGS.parent.parent / "assets"


def read_metric(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse un fichier metric MLflow (format : `timestamp value step` par ligne)."""
    steps, values = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        _, value, step = line.split()
        steps.append(int(step))
        values.append(float(value))
    return np.array(steps), np.array(values)


def find_latest_run() -> Path:
    """Renvoie le dossier du run MLflow le plus récent."""
    mlruns = config.OUTPUTS_LOGS / "mlruns"
    candidates = []
    for exp_dir in mlruns.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in (".trash", "models"):
            continue
        for run_dir in exp_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "metrics" / "loss").exists():
                candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError("Aucun run MLflow trouvé avec metrics/loss")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_loss(steps: np.ndarray, loss: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)

    # Loss brute (très bruitée car 15 images, batch=1)
    ax.plot(steps, loss, color="#94A3B8", linewidth=0.8, alpha=0.5, label="raw")

    # Moyenne glissante 50 steps (plus lisible)
    window = 50
    if len(loss) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(loss, kernel, mode="valid")
        smooth_steps = steps[window - 1:]
        ax.plot(smooth_steps, smoothed, color="#10B981", linewidth=2.2, label=f"moving avg ({window} steps)")

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("MSE loss (noise prediction)", fontsize=11)
    ax.set_title("Training loss — 800 steps, batch=1, grad_accum=4", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_lr(steps: np.ndarray, lr: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)

    ax.plot(steps, lr, color="#6366F1", linewidth=2.2)
    ax.fill_between(steps, 0, lr, color="#6366F1", alpha=0.15)

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Learning rate", fontsize=11)
    ax.set_title("Learning rate schedule — cosine with 50-step warmup", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Annotations
    peak_idx = int(np.argmax(lr))
    ax.annotate(f"peak = {lr[peak_idx]:.1e}",
                xy=(steps[peak_idx], lr[peak_idx]),
                xytext=(steps[peak_idx] + 80, lr[peak_idx] * 0.95),
                fontsize=10, color="#1E293B",
                arrowprops=dict(arrowstyle="->", color="#64748B", lw=1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    run_dir = find_latest_run()
    print(f"Run MLflow : {run_dir.relative_to(config.OUTPUTS_LOGS)}")

    loss_steps, loss_values = read_metric(run_dir / "metrics" / "loss")
    lr_steps, lr_values = read_metric(run_dir / "metrics" / "lr")

    plot_loss(loss_steps, loss_values, ASSETS_DIR / "loss_curve.svg")
    plot_lr(lr_steps, lr_values, ASSETS_DIR / "lr_schedule.svg")

    print(f"\nAssets prêts dans : {ASSETS_DIR}")


if __name__ == "__main__":
    main()
