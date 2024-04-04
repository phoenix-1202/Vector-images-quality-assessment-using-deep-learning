import zipfile
import warnings

from stata import get_distribution, normalize_images
from visualization import visualize, get_elbow_method, get_eps
from visualization import pca_visualization
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any
from clipBlip.blipModel import run_blip
from clearing import drop_duplicates, drop_unsuitable_pics
from clipBlip.clipModel import run_clip
from clustering_models import run_hdbscan, run_kmeans
from save_read import save_data, read_files
from prefect import task, flow
import configparser
from converting import rename_files
from converting import convert_to_png
import shutil
import os

warnings.filterwarnings('ignore')


@task
def get_clip_results(path, path_bad_pics):
    """получила эмбеддинги картинок"""
    dict_embs = run_clip(path)
    dict_bad_embs = run_clip(path_bad_pics)
    return dict_embs, dict_bad_embs


@task
def get_blip_results(paths_of_pics) -> list[str]:
    """получила описания к картинкам"""
    blip_model = run_blip(paths_of_pics)
    texts = blip_model.texts
    return texts


@task
def clear_data(dict_embs, dict_bad_embs) -> Dict[str, Any]:
    """чистим данные от мусора"""
    dict_embs = drop_duplicates(dict_embs)
    dict_embs = drop_unsuitable_pics(dict_embs, dict_bad_embs)
    return dict_embs


@task
def run_clustering(similarity):
    """" Получаем результаты HDBSCAN и KMEANS и выводим результаты в картинках"""
    hdbscan = run_hdbscan(similarity)
    visualize(hdbscan.best_model.labels_, similarity, path_result + 'hdbscan/distribution.png')
    pca_visualization(similarity, hdbscan.best_model.labels_, path_result + 'hdbscan/pca.png')
    get_eps(10, similarity, path_result)

    kmeans = run_kmeans(similarity)
    visualize(kmeans.best_model.labels_, similarity, path_result + 'kmeans/distribution.png')
    pca_visualization(similarity, kmeans.best_model.labels_, path_result + 'kmeans/pca.png')
    get_elbow_method(kmeans.distortions, kmeans.count_clusters_list, path_result)

    print(f"Silhouette_score KMEANS  = {kmeans.best_score}")
    print(f"Silhouette_score HDBSCAN = {hdbscan.best_score}")
    if kmeans.best_score < hdbscan.best_score:
        return hdbscan.best_model.labels_
    else:
        return kmeans.best_model.labels_


@task
def zip_folder(folder_path, output_name) -> None:
    """" Сохраняем результат: лейблы и эмбеддинги"""
    with zipfile.ZipFile(output_name, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".npz"):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))


def copy_clear_images(path_images, clear_path) -> None:
    """Перебираем каждый файл и копируем его в целевую папку"""
    for source_file in path_images:
        filename = os.path.basename(source_file)
        target_file = os.path.join(clear_path, filename)
        shutil.copyfile(source_file, target_file)


@flow()
def start(path_data, path_result, path_statistics):
    # определяю пути
    path_dirty_pics = path_data + 'svg_data/'
    path_bad_pics = path_data + 'bad_data/'
    path = path_data + 'png_data/'
    clear_path = path_data + 'clear_png_data/'

    # переименовываю и конвертирую картинки
    rename_files(path_dirty_pics)
    convert_to_png(path_dirty_pics, path)

    # чекаем распределение картинок
    get_distribution(path, path_statistics + 'distribution_first')

    # нормализуем изображения
    list_width, list_height = normalize_images(path, path_statistics + 'distribution_draft')
    save_data(list_width, path_result + 'dimensions_pictures/list_width.pkl')
    save_data(list_height, path_result + 'dimensions_pictures/list_height.pkl')

    # получаем эмбеддинги
    dict_embs, dict_bad_embs = get_clip_results(path, path_bad_pics)
    save_data(dict_embs, path_result + 'embs/embs.pkl')
    save_data(dict_bad_embs, path_result + 'embs/bad_embs.pkl')

    dict_embs = read_files(path_result + 'embs/embs.pkl')
    dict_bad_embs = read_files(path_result + 'embs/bad_embs.pkl')

    # чистим эмбеддинги картинок
    dict_embs = clear_data(dict_embs, dict_bad_embs)
    save_data(dict_embs, path_result + 'embs/clear_embs.pkl')
    clear_image_embs = read_files(path_result + 'embs/clear_embs.pkl')

    # заспукаем кластеризацию
    keys = sorted(clear_image_embs.keys())
    embeddings = [clear_image_embs[key].flatten() for key in keys]
    similarity = cosine_similarity(embeddings, embeddings)
    labels = run_clustering(similarity)
    save_data(labels, path_result + 'best_labels.txt')

    # получаем описания к картинкам
    texts = run_blip(keys)
    save_data(texts, path_result + 'texts.pkl')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('paths.txt', encoding='utf-8')

    path_data = config.get('Paths', 'path_data')
    path_statistics = config.get('Paths', 'path_statistics')
    path_result = config.get('Paths', 'path_result')
    path_visualization = config.get('Paths', 'path_visualization')

    start(path_data, path_result, path_statistics)
