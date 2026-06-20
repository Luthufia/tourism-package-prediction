import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# Dataset repository name
repo_id = "Luthufia/tourism-package-prediction"
repo_type = "dataset"

# Create the dataset repository if it does not already exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating repository...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=HF_TOKEN
    )

    print(f"Dataset repository '{repo_id}' created successfully.")

# Upload all files from the data folder
api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Dataset uploaded successfully.")
