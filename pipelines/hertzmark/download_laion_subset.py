import os
import zipfile
import shutil

def download_filmset():
    # Ensure directory exists
    os.makedirs("data", exist_ok=True)

    print("Downloading FilmSet from Kaggle…")
    os.system(
        "kaggle datasets download -d xuhangc/filmset -p data/ --force"
    )

    zip_path = "data/filmset.zip"
    if not os.path.exists(zip_path):
        raise Exception("Download failed — check Kaggle API setup.")

    print("Extracting…")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("data/filmset_raw/")

    # Move all jpg/png into data/filmset/
    dst = "data/filmset/"
    os.makedirs(dst, exist_ok=True)

    for root, dirs, files in os.walk("data/filmset_raw/"):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.move(os.path.join(root, f),
                            os.path.join(dst, f))

    print("Cleaning up…")
    shutil.rmtree("data/filmset_raw/")
    os.remove(zip_path)

    print("FilmSet downloaded to: data/filmset/")
    print("Total images:", len(os.listdir(dst)))

if __name__ == "__main__":
    download_filmset()
