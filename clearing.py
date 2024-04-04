from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import torch
from skimage.transform import resize
from skimage import io
from skimage.metrics import structural_similarity as ssim
import os


def check_ssim(path1, path2):
    try:
        image1 = io.imread(path1, as_gray=True)
        image2 = io.imread(path2, as_gray=True)
    except ValueError:
        return False

    new_shape = (
        max(image1.shape[0], image2.shape[0]),
        max(image1.shape[1], image2.shape[1])
    )

    image1 = resize(image1, new_shape, anti_aliasing=True)
    image2 = resize(image2, new_shape, anti_aliasing=True)

    r_range = image1.max() - image1.min()
    q_range = image2.max() - image2.min()
    diff = ssim(image1, image2, channel_axis=-1, data_range=max(r_range, q_range))
    if diff > 0.95:
        return True, diff
    else:
        return False, diff


def drop_duplicates(dict_all_embs):
    """Удаляем дубликаты"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dict_copy = dict_all_embs.copy()
    for key1, emb1 in tqdm(dict_copy.items(), desc="Удаляем дубликаты"):
        for key2, emb2 in dict_copy.items():
            if key1 != key2:
                if key1 in dict_all_embs.keys() and key2 in dict_all_embs.keys():
                    ans, diff = check_ssim(key1, key2)
                    if ans:
                        del dict_all_embs[key2]
                # if cosine_similarity(emb1.to(device),
                #                      emb2.to(device)) > 0.99999999999999999 and key2 in dict_all_embs.keys():
                #     del dict_all_embs[key2]

    return dict_all_embs


def drop_unsuitable_pics(dict_all_embs, dict_bad_embs):
    """Здесь удаляем неподходящие картинки (типа китайские надписи и другая шелуха)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dict_copy = dict_all_embs.copy()
    for key1, emb1 in tqdm(dict_copy.items(), desc="Удаляем неподходящие картинки"):
        for key2, emb2 in dict_bad_embs.items():
            if key1 in dict_all_embs.keys():
                ans, diff = check_ssim(key1, key2)
                if ans:
                    del dict_all_embs[key1]
                # if cosine_similarity(emb1.to(device),
                #                      emb2.to(device)) > 0.99999999999999999 and key2 in dict_all_embs.keys():
                #     del dict_all_embs[key1]

    return dict_all_embs
