from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths
    DATA_RAW: Path = ROOT_DIR / "data" / "1-raw"
    DATA_PROCESSED: Path = ROOT_DIR / "data" / "2-processed"
    DATA_EXTERNAL: Path = ROOT_DIR / "data" / "3-external"
    OUTPUTS_MODELS: Path = ROOT_DIR / "outputs" / "models"
    OUTPUTS_LOGS: Path = ROOT_DIR / "outputs" / "logs"
    OUTPUTS_RESULTS: Path = ROOT_DIR / "outputs" / "results"

    # Dataset
    DRAWINGS_DIR: Path = ROOT_DIR / "data" / "1-raw" / "my_drawings"
    DATASET_DIR: Path = ROOT_DIR / "data" / "2-processed" / "thibchibi_dataset"
    IMAGE_SIZE: int = 512

    # Reproducibility
    RANDOM_STATE: int = 42

    # ----- Stable Diffusion + LoRA -----
    # Mirror communautaire de SD 1.5 (l'org "runwayml" a été retirée mi-2024).
    BASE_MODEL: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    TRIGGER_WORD: str = "thibchibi"  # token unique pour invoquer le style appris
    LORA_OUTPUT_DIR: Path = ROOT_DIR / "outputs" / "models" / "lora_thibchibi"

    # Hyperparamètres LoRA
    LORA_RANK: int = 16
    LORA_ALPHA: int = 16

    # Hyperparamètres entraînement (réglés pour GTX 1660 Ti 6 Go)
    LEARNING_RATE: float = 1e-4
    MAX_TRAIN_STEPS: int = 800
    TRAIN_BATCH_SIZE: int = 1
    GRADIENT_ACCUMULATION_STEPS: int = 4
    MIXED_PRECISION: str = "fp16"
    GRADIENT_CHECKPOINTING: bool = True
    SEED: int = 42

    # Inférence
    INFERENCE_STEPS: int = 30
    GUIDANCE_SCALE: float = 7.5

    # MLflow
    MLFLOW_EXPERIMENT_NAME: str = "thibchibi_lora"


config = Config()
