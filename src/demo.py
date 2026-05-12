"""Mini-app Gradio pour tester le LoRA thibchibi interactivement.

Lancement :
    python -m src.demo
Puis ouvrir http://127.0.0.1:7860 dans le navigateur.

Le pipeline SD + LoRA est chargé UNE SEULE FOIS au démarrage (sinon ~4 Go re-chargés à chaque clic).
Le `lora_scale` peut être ajusté à la volée via `cross_attention_kwargs` sans recharger.
"""

import random

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

from src.config import config


DEFAULT_NEGATIVE_PROMPT = "blurry, lowres, deformed, ugly, bad anatomy"


def load_pipeline() -> StableDiffusionPipeline:
    """Charge SD 1.5 + LoRA une fois au démarrage."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Chargement du pipeline ({config.BASE_MODEL}) sur {device} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    if not (config.LORA_OUTPUT_DIR / "pytorch_lora_weights.safetensors").exists():
        raise FileNotFoundError(
            f"LoRA introuvable : {config.LORA_OUTPUT_DIR}\n"
            f"Lance d'abord : python -m src.models.train"
        )
    pipe.load_lora_weights(str(config.LORA_OUTPUT_DIR))
    # On NE fuse PAS : on passera le scale via cross_attention_kwargs à chaque appel
    # (permet de changer lora_scale dynamiquement sans recharger).

    pipe.enable_vae_slicing()
    print("Pipeline prêt.")
    return pipe


PIPE = load_pipeline()


def generate(
    prompt: str,
    negative_prompt: str,
    num_images: int,
    steps: int,
    guidance_scale: float,
    lora_scale: float,
    seed: int,
):
    if not prompt.strip():
        raise gr.Error("Le prompt ne peut pas être vide.")

    # Seed -1 -> aléatoire
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=PIPE.device).manual_seed(int(seed))

    images = PIPE(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_images_per_prompt=int(num_images),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        cross_attention_kwargs={"scale": float(lora_scale)},
        generator=generator,
    ).images

    info = (
        f"Seed : {seed} | LoRA scale : {lora_scale} | Steps : {steps} | "
        f"Guidance : {guidance_scale} | Negative : {negative_prompt or '—'}"
    )
    return images, info


def randomize_seed() -> int:
    return random.randint(0, 2**32 - 1)


with gr.Blocks(title="thibchibi LoRA — demo") as demo:
    gr.Markdown(
        f"# thibchibi LoRA demo\n"
        f"Fine-tuning de SD 1.5 sur les dessins chibi. "
        f"Trigger word : **`{config.TRIGGER_WORD}`** (inclus-le dans ton prompt pour invoquer le style)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value=f"a {config.TRIGGER_WORD} chibi character, waving hello",
                lines=2,
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                value=DEFAULT_NEGATIVE_PROMPT,
                lines=1,
            )

            with gr.Row():
                num_images = gr.Slider(1, 4, value=2, step=1, label="Nombre d'images")
                steps = gr.Slider(10, 50, value=config.INFERENCE_STEPS, step=1, label="Steps")

            with gr.Row():
                guidance_scale = gr.Slider(1.0, 15.0, value=config.GUIDANCE_SCALE, step=0.5, label="Guidance scale")
                lora_scale = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="LoRA scale")

            with gr.Row():
                seed = gr.Number(value=-1, label="Seed (-1 = aléatoire)", precision=0)
                randomize_btn = gr.Button("🎲 Random seed", scale=0)

            generate_btn = gr.Button("Générer", variant="primary")

        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Sorties", columns=2, height=512)
            info = gr.Textbox(label="Info", interactive=False)

    randomize_btn.click(fn=randomize_seed, outputs=seed)
    generate_btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, num_images, steps, guidance_scale, lora_scale, seed],
        outputs=[gallery, info],
    )

    gr.Examples(
        examples=[
            [f"a {config.TRIGGER_WORD} chibi character, waving hello in welcome"],
            [f"a {config.TRIGGER_WORD} chibi character, with arms crossed wearing sunglasses"],
            [f"a {config.TRIGGER_WORD} chibi character, riding a bike, sunny day"],
            [f"a {config.TRIGGER_WORD} chibi character, drinking coffee, cozy cafe background"],
            [f"a {config.TRIGGER_WORD} chibi character, looking surprised, big eyes"],
        ],
        inputs=prompt,
    )


if __name__ == "__main__":
    demo.launch()
