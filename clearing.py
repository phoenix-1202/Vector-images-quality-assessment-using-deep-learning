from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import torch
from skimage.transform import resize
from skimage import io
from skimage.metrics import structural_similarity as ssim
import os
import pandas as pd
import faiss
import numpy as np


def drop_duplicates(path_png, embeddings):
    X = np.concatenate(embeddings, axis=0)
    df = pd.DataFrame()
    df["file_path"] = [os.path.join(path_png, i) for i in os.listdir(path_png)]
    df["id"] = [i for i in range(len(df))]
    df["embedding"] = [embed for embed in X]

    d = 1152
    index = faiss.IndexFlatL2(d)
    index.add(X)
    batch_size = 1000
    df_batches = [df.iloc[i:i + batch_size].copy() for i in range(0, len(df.id.tolist()), batch_size)]
    res_dfs = []
    for batch in tqdm(df_batches):
        distances, indexes = index.search(np.array(batch.embedding.tolist()), 30)
        batch["duplicates"] = [list(zip(dist, idx)) for dist, idx in zip(distances, indexes)]
        res_dfs.append(batch)

    full_df = pd.concat(res_dfs)
    full_df.reset_index(drop=True, inplace=True)

    threshold = 10

    remove_items = set()
    keep_items = set()
    abobus = []
    full_df.reset_index(drop=True, inplace=True)

    for idx, raw in tqdm(full_df.iterrows(), total=len(full_df)):
        if not (idx in remove_items):
            keep_items.add(idx)
        a = [(idx, -100)]
        for i_dist, i_idx in raw["duplicates"]:
            if i_dist < threshold:
                a.append((i_dist, i_idx))
            if i_dist < threshold and not (i_idx in keep_items):
                remove_items.add(i_idx)
        abobus.append(a)
    train_df = full_df.loc[~full_df.index.isin((list(remove_items)))]
    result_dict = train_df.set_index('file_path')['embedding'].to_dict()
    print(f"REMOVED {len(remove_items)}")
    return result_dict


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
