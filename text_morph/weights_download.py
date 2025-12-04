import argparse
from huggingface_hub import snapshot_download

# Hugging Face token
HF_TOKEN = "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"

def main():
    parser = argparse.ArgumentParser(description="Download model/dataset from Hugging Face Hub")
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Root path where the model/dataset will be downloaded (default: current directory)"
    )
    
    args = parser.parse_args()
    
    print(f"Downloading to: {args.path}")
    
    # Download the entire repository
    snapshot_download(
        repo_id="MrBosch/text_ctrl_weights",
        repo_type="model",
        local_dir=args.path,
        token=HF_TOKEN
    )
    
    print("Download completed successfully!")

if __name__ == "__main__":
    main()

# python download.py --path /your/custom/path