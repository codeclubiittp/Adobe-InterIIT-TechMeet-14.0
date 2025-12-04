# embed_inference.py
import cv2
import torch
from models import Embedder

def embed_image(image_path, out_path, key_seed=1234):
    device = "cuda"

    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    img_t = torch.tensor(img).float().permute(2,0,1)/255.0
    img_t = img_t.unsqueeze(0).to(device)

    embedder = Embedder().to(device)
    embedder.load_state_dict(torch.load("embedder.pth"))

    torch.manual_seed(key_seed)
    key_vec = torch.randn(1, 128).to(device)

    wm = embedder(img_t, key_vec)[0]
    wm_np = (wm.permute(1,2,0).cpu().detach().numpy() * 255).astype("uint8")

    cv2.imwrite(out_path, wm_np)
    print("Watermarked image saved:", out_path)

if __name__ == "__main__":
    embed_image("data/filmset/_DSF3769.png", "watermarked.jpg")
