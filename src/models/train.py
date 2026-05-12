"""Boucle d'entraînement LoRA Stable Diffusion 1.5 + logging MLflow.

Pipeline :
    1. Charge les composants SD 1.5 (tokenizer, text_encoder, vae, unet, scheduler).
    2. Gèle tous les paramètres, puis attache des adapters LoRA au UNet via peft.
    3. Itère sur le dataset (DataLoader): VAE encode -> ajoute bruit -> UNet prédit bruit -> MSE.
    4. Sauvegarde l'adapter LoRA au format diffusers (chargeable avec pipe.load_lora_weights).

Note: avec 15 images pour un style transfer, on n'a pas de split train/val classique.
L'évaluation se fait visuellement après training via `src/models/generate.py`.
"""

import shutil
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import get_peft_model_state_dict
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from src.config import config
from src.models.model import attach_lora_to_unet


def build_dataloader(tokenizer: CLIPTokenizer) -> DataLoader:
    """Charge le dataset HF imagefolder + applique transforms image + tokenize captions."""
    if not (config.DATASET_DIR / "metadata.jsonl").exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {config.DATASET_DIR}\n"
            f"Lance d'abord : python -m src.data.make_dataset"
        )

    dataset = load_dataset("imagefolder", data_dir=str(config.DATASET_DIR), split="train")

    image_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # -> [-1, 1] pour la VAE
    ])

    def collate_fn(batch):
        pixel_values = torch.stack([image_transforms(b["image"].convert("RGB")) for b in batch])
        captions = [b["text"] for b in batch]
        input_ids = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    dataloader = DataLoader(
        dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows + petit dataset -> 0 worker plus simple
    )
    return dataloader, len(dataset)


def save_lora(unet: UNet2DConditionModel, output_dir: Path) -> None:
    """Sauvegarde l'adapter LoRA au format diffusers (safetensors)."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
    )


def main() -> None:
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    else:
        raise RuntimeError("CUDA n'est pas disponible — entraînement LoRA SD requis sur GPU.")

    accelerator = Accelerator(
        mixed_precision=config.MIXED_PRECISION,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    )
    device = accelerator.device
    weight_dtype = torch.float16 if config.MIXED_PRECISION == "fp16" else torch.float32

    # ----- Composants SD 1.5 -----
    print(f"Chargement de {config.BASE_MODEL} ...")
    tokenizer = CLIPTokenizer.from_pretrained(config.BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.BASE_MODEL, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.BASE_MODEL, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.BASE_MODEL, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.BASE_MODEL, subfolder="scheduler")

    # Tout est frozen + .eval() pour désactiver tout dropout/BN éventuel
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # GPU + fp16 pour les modules frozen (économise VRAM)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # ----- LoRA -----
    # IMPORTANT : add_adapter AVANT enable_gradient_checkpointing pour que le
    # checkpoint context voie les params trainables (les LoRA).
    attach_lora_to_unet(unet, rank=config.LORA_RANK, alpha=config.LORA_ALPHA)

    if config.GRADIENT_CHECKPOINTING:
        unet.enable_gradient_checkpointing()
    # On force les params LoRA en fp32 pour la stabilité numérique (gradient en fp32)
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"Paramètres LoRA entraînables : {n_trainable:,} ({100 * n_trainable / n_total:.2f}% du UNet)")

    # ----- Data -----
    dataloader, dataset_size = build_dataloader(tokenizer)

    # ----- Optim -----
    optimizer = torch.optim.AdamW(lora_params, lr=config.LEARNING_RATE)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=config.MAX_TRAIN_STEPS,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # ----- MLflow -----
    mlruns_dir = config.OUTPUTS_LOGS / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir.as_posix()}")
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    print(f"\nTraining {config.MAX_TRAIN_STEPS} steps | {dataset_size} images | device={device}")
    print(f"Mixed precision={config.MIXED_PRECISION} | grad_accum={config.GRADIENT_ACCUMULATION_STEPS}\n")

    with mlflow.start_run():
        mlflow.log_params({
            "base_model": config.BASE_MODEL,
            "trigger_word": config.TRIGGER_WORD,
            "lora_rank": config.LORA_RANK,
            "lora_alpha": config.LORA_ALPHA,
            "learning_rate": config.LEARNING_RATE,
            "max_steps": config.MAX_TRAIN_STEPS,
            "batch_size": config.TRAIN_BATCH_SIZE,
            "grad_accumulation": config.GRADIENT_ACCUMULATION_STEPS,
            "image_size": config.IMAGE_SIZE,
            "dataset_size": dataset_size,
            "seed": config.SEED,
        })

        global_step = 0
        unet.train()
        dataloader_iter = iter(dataloader)

        while global_step < config.MAX_TRAIN_STEPS:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            with accelerator.accumulate(unet):
                # Image -> latents (VAE encode)
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                # Bruit + timesteps aléatoires
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Embeddings texte
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]

                # Prédiction de bruit par le UNet (+ LoRA)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                loss_val = loss.detach().item()
                lr_val = lr_scheduler.get_last_lr()[0]
                if global_step % 10 == 0 or global_step == 1:
                    print(f"  step {global_step:4d}/{config.MAX_TRAIN_STEPS}  |  loss={loss_val:.4f}  |  lr={lr_val:.2e}")
                mlflow.log_metric("loss", loss_val, step=global_step)
                mlflow.log_metric("lr", lr_val, step=global_step)

        # ----- Save -----
        unwrapped_unet = accelerator.unwrap_model(unet)
        save_lora(unwrapped_unet, config.LORA_OUTPUT_DIR)
        print(f"\nLoRA sauvegardé -> {config.LORA_OUTPUT_DIR}")
        mlflow.log_artifacts(str(config.LORA_OUTPUT_DIR), artifact_path="lora")


if __name__ == "__main__":
    main()
