"""Robust Oxford-IIIT Pet dataset download with retry + resume."""
import os
import sys
import time
import urllib.request
import tarfile

DATA_ROOT = "D:/datasets/oxford-iiit-pet"
FILES = [
    ("images.tar.gz", "https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz"),
    ("annotations.tar.gz", "https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz"),
]
MAX_RETRIES = 20
RETRY_DELAY = 60  # seconds between retries


def download_with_resume(url: str, dest: str) -> None:
    """Download a file with retry logic. Shows progress."""
    temp = dest + ".part"

    # Check existing partial download
    existing = os.path.getsize(temp) if os.path.exists(temp) else 0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                print(f"  Resuming from {existing / 1024 / 1024:.1f}MB")

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                total = int(resp.headers.get("Content-Length", 0)) + existing
                mode = "ab" if existing > 0 else "wb"

                with open(temp, mode) as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        existing += len(chunk)
                        if total:
                            pct = existing * 100 // total
                            mb = existing / 1024 / 1024
                            print(f"\r  {pct}% ({mb:.1f}MB) - attempt {attempt}", end="", flush=True)

            if total > 0:
                print()
            os.rename(temp, dest)
            return  # success

        except Exception as e:
            print(f"\n  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            existing = os.path.getsize(temp) if os.path.exists(temp) else 0

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts")


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)

    for filename, url in FILES:
        dest = os.path.join(DATA_ROOT, filename)
        if os.path.exists(dest):
            print(f"{filename} already downloaded, skipping")
            continue

        print(f"Downloading {filename} ({url})")
        download_with_resume(url, dest)

        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  Downloaded: {size_mb:.1f}MB")

        # Extract
        print(f"  Extracting {filename}...")
        with tarfile.open(dest) as tar:
            tar.extractall(path=DATA_ROOT, filter="data")
        os.remove(dest)
        print(f"  Extracted and cleaned up.")

    print(f"\nAll done! Contents of {DATA_ROOT}:")
    for item in sorted(os.listdir(DATA_ROOT)):
        print(f"  {item}")


if __name__ == "__main__":
    main()
