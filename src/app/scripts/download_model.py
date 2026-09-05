from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    model_dir = Path(__file__).resolve().parents[3] / "models" / "finbert"
    snapshot_download(repo_id="ProsusAI/finbert", local_dir=model_dir)
    print(f"Downloaded to {model_dir}")


if __name__ == "__main__":
    main()
