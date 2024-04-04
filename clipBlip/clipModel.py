import warnings

warnings.filterwarnings('ignore')
import torch
from PIL import Image
import clip
import os


class ClipModel(object):
    def __init__(self):
        self.embeddings = {}

    def get_embs(self, pathPic=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelCLIP, preprocess = clip.load("ViT-B/32", device=device)
        embeddings = {}
        for filename in os.listdir(pathPic):
            path_file = pathPic + filename
            if path_file.endswith(".ipynb_checkpoints"):
                continue
            image = preprocess(Image.open(path_file)).unsqueeze(0).to(device)
            with torch.no_grad():
                one_emb = modelCLIP.encode_image(image)
                embeddings[path_file] = one_emb

        return embeddings

    def update_fields(self, path):
        self.embeddings = self.get_embs(path)


def run_clip(path):
    my_clip = ClipModel()
    my_clip.update_fields(path)
    return my_clip.embeddings
