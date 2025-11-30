# scripts/download_finbert.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ProsusAI/finbert",
    local_dir="models/finbert",
    local_dir_use_symlinks=False,
)
print("Downloaded to models/finbert")
