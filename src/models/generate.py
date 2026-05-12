"""Inférence : charge SD 1.5 + LoRA thibchibi et génère des images depuis un prompt.

Usage :
    python -m src.models.generate --prompt "a thibchibi character riding a bike"
    python -m src.models.generate --prompt "..." --num-images 4 --seed 123 --lora-scale 0.8
"""

import argparse
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline

from src.config import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Génère des images via SD 1.5 + LoRA thibchibi.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt texte (inclure 'thibchibi' pour invoquer le style).")
    parser.add_argument("--negative-prompt", type=str, default="blurry, lowres, deformed, ugly, bad anatomy")
    parser.add_argument("--num-images", type=int, default=4, help="Nombre d'images à générer.")
    parser.add_argument("--steps", type=int, default=config.INFERENCE_STEPS, help="Nombre de denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=config.GUIDANCE_SCALE, help="Classifier-free guidance scale.")
    parser.add_argument("--lora-scale", type=float, default=0.8, help="Force du LoRA (0.0 = SD vanilla, 1.0 = full LoRA).")
    parser.add_argument("--seed", type=int, default=None, help="Seed pour la reproductibilité (par défaut: aléatoire).")
    parser.add_argument("--no-lora", action="store_true", help="Génère avec SD 1.5 vanilla (sans LoRA) pour comparaison.")
    return parser.parse_args()


def select_dtype(device: str) -> torch.dtype:
    """fp16 sur Ampere+ (SM >= 8.0), fp32 sur Turing/older (évite les NaN)."""
    if device != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.float16 if major >= 8 else torch.float32


def build_pipeline(use_lora: bool, lora_scale: float) -> StableDiffusionPipeline:
    """Construit le pipeline SD 1.5, charge le LoRA si demandé."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,  # désactive le filtre NSFW (sinon nos chibis pourraient être blanked sur false positive)
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    if use_lora:
        lora_path = config.LORA_OUTPUT_DIR
        if not (lora_path / "pytorch_lora_weights.safetensors").exists():
            raise FileNotFoundError(
                f"LoRA introuvable : {lora_path}\n"
                f"Lance d'abord : python -m src.models.train"
            )
        pipe.load_lora_weights(str(lora_path))
        pipe.fuse_lora(lora_scale=lora_scale)
        print(f"LoRA chargé depuis {lora_path} (scale={lora_scale})")

    # Optims VRAM (inférence)
    pipe.enable_vae_slicing()
    return pipe


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(use_lora=not args.no_lora, lora_scale=args.lora_scale)

    seed = args.seed if args.seed is not None else torch.seed() & 0xFFFFFFFF
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    print(f"\nPrompt          : {args.prompt}")
    print(f"Negative prompt : {args.negative_prompt}")
    print(f"Steps           : {args.steps} | Guidance : {args.guidance_scale} | Seed : {seed}")
    print(f"LoRA            : {'OFF (vanilla SD)' if args.no_lora else f'ON (scale={args.lora_scale})'}\n")

    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images

    # ----- Save -----
    config.OUTPUTS_RESULTS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "vanilla" if args.no_lora else f"lora{args.lora_scale:.1f}"
    run_dir = config.OUTPUTS_RESULTS / f"{timestamp}_{tag}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        img.save(run_dir / f"image_{i:02d}.png")

    # Petit .txt avec les params (pour ne pas oublier ce qu'on a généré)
    (run_dir / "prompt.txt").write_text(
        f"prompt: {args.prompt}\n"
        f"negative_prompt: {args.negative_prompt}\n"
        f"steps: {args.steps}\n"
        f"guidance_scale: {args.guidance_scale}\n"
        f"lora_scale: {0.0 if args.no_lora else args.lora_scale}\n"
        f"seed: {seed}\n"
        f"base_model: {config.BASE_MODEL}\n",
        encoding="utf-8",
    )

    print(f"{len(images)} image(s) sauvegardée(s) -> {run_dir}")


if __name__ == "__main__":
    main()
