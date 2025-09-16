import os
import subprocess

def download_kaggle_dataset(download_path: str = "data/external", unzip: bool = True):
    """
    Downloads the 'reddybharat/pd-augmented' (private) dataset from Kaggle using the Kaggle API.

    Args:
        download_path (str): Directory to download the dataset to.
        unzip (bool): Whether to unzip the dataset after download.

    Raises:
        RuntimeError: If the Kaggle API is not installed or download fails.
    """
    # Ensure Kaggle API is installed
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise RuntimeError("Kaggle package not installed. Please install it with 'pip install kaggle'.")

    # Ensure Kaggle API credentials are set
    if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        raise RuntimeError("Kaggle API credentials not found. Please place your kaggle.json in ~/.kaggle/.")

    os.makedirs(download_path, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download", "-d", "reddybharat/pd-augmented", "-p", download_path
    ]
    if unzip:
        cmd.append("--unzip")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaggle download failed: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download the 'reddybharat/pd-augmented' dataset from Kaggle.")
    parser.add_argument('--download_path', type=str, default='data/external', help='Directory to download the dataset to.')
    parser.add_argument('--unzip', action='store_true', help='Unzip the dataset after download.')
    parser.add_argument('--no-unzip', dest='unzip', action='store_false', help='Do not unzip the dataset after download.')
    parser.set_defaults(unzip=True)
    args = parser.parse_args()
    try:
        download_kaggle_dataset(download_path=args.download_path, unzip=args.unzip)
        print(f"Dataset downloaded successfully to {args.download_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
