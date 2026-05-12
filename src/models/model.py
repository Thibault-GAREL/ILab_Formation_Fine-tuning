"""Architecture du modèle (sans logique d'entraînement).

Helpers pour construire le UNet Stable Diffusion + LoRA adapters.
"""

from diffusers import UNet2DConditionModel
from peft import LoraConfig


# Modules d'attention ciblés par les LoRA dans le UNet de SD 1.5.
# (cross-attention text<->image dans les blocs down/mid/up)
UNET_LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def attach_lora_to_unet(unet: UNet2DConditionModel, rank: int, alpha: int) -> UNet2DConditionModel:
    """Injecte des adapters LoRA dans les couches d'attention du UNet.

    Le UNet doit déjà être chargé et avoir ses paramètres frozen (requires_grad=False).
    Après cet appel, seuls les paramètres LoRA seront entraînables.
    """
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=UNET_LORA_TARGET_MODULES,
    )
    unet.add_adapter(lora_config)
    return unet
