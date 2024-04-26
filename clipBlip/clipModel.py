import warnings

warnings.filterwarnings('ignore')
import torch
from PIL import Image
import clip
import os
from tqdm import tqdm
from open_clip import create_model_from_pretrained


def get_vit_embs(path_png):
    device = "cuda:0"
    dict_all_embs = {}
    model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP')
    # model.to(device)

    for filename in tqdm(os.listdir(path_png), desc="Берем ембеддинги"):
        image_path = os.path.join(path_png, filename)
        image = Image.open(image_path).convert('RGB')
        inputs = preprocess(image).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            # features = model.encode_image(inputs.to(device))
            features = model.encode_image(inputs)
        embeds = features.detach().cpu().numpy()
        dict_all_embs[image_path] = embeds

    return dict_all_embs


class ClipModel(object):
    def __init__(self):
        self.embeddings = {}

    def get_embs(self, pathPic=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelCLIP, preprocess = clip.load("ViT-B/32", device=device)
        embeddings = {}
        for filename in tqdm(os.listdir(pathPic), desc="Берем эмбдеддинги"):
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
