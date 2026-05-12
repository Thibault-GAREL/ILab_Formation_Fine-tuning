"""Pipeline de données : load PNG -> resize carré 512 -> captions auto -> dataset HF.

Lit les dessins bruts dans `data/1-raw/my_drawings/`, génère une caption pour chaque
image à partir de son nom de fichier (français -> anglais), redimensionne en 512x512
avec padding blanc, et écrit un dataset au format ImageFolder + metadata.jsonl
dans `data/2-processed/thibchibi_dataset/` — directement consommable par
🤗 datasets.load_dataset("imagefolder", data_dir=...).
"""

import json
import shutil
from pathlib import Path

from PIL import Image

from src.config import config


POSTURE_KEYWORDS: list[tuple[str, str]] = [
    ("bras croisé lunette de soleil", "with arms crossed wearing sunglasses"),
    ("bras croisés yeux fermés", "with arms crossed and eyes closed"),
    ("bras croisés", "with arms crossed"),
    ("bras croisé", "with arms crossed"),
    ("doigt en l'air", "with one finger raised up"),
    ("au revoir", "waving goodbye"),
    ("bienvenu", "waving hello in welcome"),
    ("montre", "pointing at something"),
    ("attend", "waiting patiently"),
    ("surprise", "looking surprised"),
]


def caption_from_filename(filename: str, trigger: str) -> str:
    """Mappe un nom de fichier (français, avec variantes numérotées) vers une caption EN."""
    name = filename.lower().replace("capture", "").replace(".png", "").strip()
    for keyword, description in POSTURE_KEYWORDS:
        if keyword in name:
            return f"a {trigger} chibi character, {description}"
    return f"a {trigger} chibi character"


def resize_square(img: Image.Image, size: int, background: tuple = (255, 255, 255)) -> Image.Image:
    """Conserve le ratio, padding blanc pour obtenir une image carrée size×size."""
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), background)
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def main() -> None:
    raw_dir: Path = config.DRAWINGS_DIR
    out_dir: Path = config.DATASET_DIR

    if not raw_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {raw_dir}")

    images = sorted({p for p in raw_dir.iterdir() if p.suffix.lower() == ".png"})
    if not images:
        raise FileNotFoundError(f"Aucun PNG trouvé dans {raw_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for i, img_path in enumerate(images):
        new_name = f"{i:04d}.png"
        img = resize_square(Image.open(img_path), config.IMAGE_SIZE)
        img.save(out_dir / new_name)

        caption = caption_from_filename(img_path.name, config.TRIGGER_WORD)
        metadata.append({"file_name": new_name, "text": caption})
        print(f"  [{i + 1:02d}/{len(images)}] {img_path.name:<45s} -> {new_name}  |  {caption}")

    metadata_path = out_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDataset prêt : {len(metadata)} images -> {out_dir}")
    print(f"Metadata    : {metadata_path}")


if __name__ == "__main__":
    main()
