from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from tqdm import tqdm


class BlipModel(object):

    def __init__(self):
        self.paths = []
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.modelBLIP = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.texts = []

    def get_texts(self, paths):
        self.texts = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelBLIP.to(device)
        for filename in tqdm(self.paths, desc="Генерим описания"):
            image = Image.open(filename).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(device, torch.float16)
            generated_ids = self.modelBLIP.generate(**inputs, max_new_tokens=7)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            self.texts.append(generated_text)

        return self.texts


def run_blip(paths):
    print("run blip")
    model_blip = BlipModel()
    return model_blip.get_texts(paths)
