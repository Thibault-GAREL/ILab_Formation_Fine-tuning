"""Réécrit les `artifact_uri` absolus dans tous les meta.yaml de outputs/logs/mlruns/.

Utile après avoir transféré un run MLflow d'une autre machine (ex: RunPod Linux -> PC Windows).
MLflow file-store stocke des chemins absolus, qui cassent le UI quand on déplace le dossier.

Usage : python -m src.fix_mlflow_paths
"""

from pathlib import Path

from src.config import config


def fix_meta_files() -> None:
    mlruns_dir = config.OUTPUTS_LOGS / "mlruns"
    if not mlruns_dir.exists():
        print(f"Aucun dossier mlruns trouvé : {mlruns_dir}")
        return

    target_uri_prefix = (mlruns_dir).resolve().as_uri()  # ex: file:///D:/.../outputs/logs/mlruns
    fixed = 0

    for meta in mlruns_dir.rglob("meta.yaml"):
        content = meta.read_text(encoding="utf-8")
        new_content = []
        changed = False
        for line in content.splitlines():
            if line.startswith("artifact_uri:") or line.startswith("artifact_location:"):
                key, _, _ = line.partition(":")
                rel = meta.parent.relative_to(mlruns_dir).as_posix()
                if rel in (".", ""):
                    new_uri = target_uri_prefix
                else:
                    new_uri = f"{target_uri_prefix}/{rel}/artifacts" if key == "artifact_uri" else f"{target_uri_prefix}/{rel}"
                new_line = f"{key}: {new_uri}"
                if new_line != line:
                    changed = True
                new_content.append(new_line)
            else:
                new_content.append(line)
        if changed:
            meta.write_text("\n".join(new_content) + "\n", encoding="utf-8")
            print(f"Fixed : {meta.relative_to(mlruns_dir)}")
            fixed += 1

    print(f"\n{fixed} fichier(s) meta.yaml corrigé(s) -> {mlruns_dir}")
    print("Tu peux maintenant lancer : mlflow ui --backend-store-uri file:./outputs/logs/mlruns")


if __name__ == "__main__":
    fix_meta_files()
