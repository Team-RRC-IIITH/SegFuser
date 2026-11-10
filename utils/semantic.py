import torch
import torch.nn.functional as F
from PIL import Image
import clip

DEVICE = 'cuda'

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

@torch.no_grad()
def compute_clip_embeddings(image, boxes):
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        crop = Image.fromarray(image[y1:y2, x1:x2])
        crops.append(clip_preprocess(crop))

    if not crops:
        return torch.empty((0, 512), device=DEVICE)

    crops = torch.stack(crops).to(DEVICE)
    embeddings = clip_model.encode_image(crops)
    return F.normalize(embeddings.float(), dim=-1)