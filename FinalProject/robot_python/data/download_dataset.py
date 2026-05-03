"""Download the MIT Killian CARMEN dataset if not already present."""

import os
import urllib.request

URL = "http://ais.informatik.uni-freiburg.de/slamevaluation/datasets/intel.clf"
CLF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intel.clf")


def download_dataset(path: str = CLF_PATH) -> str:
    if os.path.exists(path):
        print(f"Dataset already exists: {path}")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading Intel Research Lab dataset...")
    urllib.request.urlretrieve(URL, path)
    print(f"Saved to {path}")
    return path


if __name__ == "__main__":
    download_dataset()
