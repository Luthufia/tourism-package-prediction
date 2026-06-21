import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# --------------------------------------------------
# Hugging Face Authentication
# --------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# --------------------------------------------------
# Hugging Face Space Repository
# --------------------------------------------------

repo_id = "Luthufia/tourism-package-prediction-app-v2"

# --------------------------------------------------
# Check whether Space exists
# --------------------------------------------------

try:
    api.repo_info(
        repo_id=repo_id,
        repo_type="space"
    )

    print(f"Space '{repo_id}' already exists.")

except RepositoryNotFoundError:

    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        token=HF_TOKEN
    )

    print(f"Space '{repo_id}' created.")

# --------------------------------------------------
# Deployment Folder
# --------------------------------------------------

folder_path = "deployment"

# --------------------------------------------------
# Upload Deployment Files
# --------------------------------------------------

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="space"
)

print("Deployment files uploaded successfully.")
