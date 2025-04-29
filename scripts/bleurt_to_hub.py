import shutil
from tempfile import TemporaryDirectory

import requests
from huggingface_hub import create_repo, upload_folder
from tqdm import tqdm


CHECKPOINT_URLS = {
    "bleurt-tiny-128": "https://storage.googleapis.com/bleurt-oss/bleurt-tiny-128.zip",
    "bleurt-tiny-512": "https://storage.googleapis.com/bleurt-oss/bleurt-tiny-512.zip",
    "bleurt-base-128": "https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip",
    "bleurt-base-512": "https://storage.googleapis.com/bleurt-oss/bleurt-base-512.zip",
    "bleurt-large-128": "https://storage.googleapis.com/bleurt-oss/bleurt-large-128.zip",
    "bleurt-large-512": "https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip",
    "BLEURT-20-D3": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip",
    "BLEURT-20-D6": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D6.zip",
    "BLEURT-20-D12": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip",
    "BLEURT-20": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip",
}


def download_bleurt_checkpoints():
    for checkpoint_name, url in CHECKPOINT_URLS.items():
        with TemporaryDirectory() as temp_dir:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                progressbar = tqdm(
                    total=int(response.headers.get("content-length", 0)),
                    unit="B",
                    unit_scale=True,
                    desc=checkpoint_name,
                )
                with open(f"{temp_dir}/{checkpoint_name}.zip", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progressbar.update(len(chunk))
                progressbar.close()

                # Unzip the downloaded file (if needed)
                shutil.unpack_archive(f"{temp_dir}/{checkpoint_name}.zip", extract_dir=temp_dir)

                create_repo(
                    repo_id=f"BramVanroy/{checkpoint_name}",
                    repo_type="model",
                    exist_ok=True,
                )
                upload_folder(
                    repo_id=f"BramVanroy/{checkpoint_name}",
                    folder_path=f"{temp_dir}/{checkpoint_name}",
                    commit_message=f"Upload {checkpoint_name}",
                    repo_type="model",
                )

                print(f"Downloaded {checkpoint_name} to {temp_dir}/{checkpoint_name}.zip")
            else:
                raise Exception(
                    f"Failed to download {checkpoint_name} from {url}. Status code: {response.status_code}. Error: {response.text}"
                )


if __name__ == "__main__":
    download_bleurt_checkpoints()
    print("All BLEURT checkpoints have been downloaded and uploaded to Hugging Face Hub.")
