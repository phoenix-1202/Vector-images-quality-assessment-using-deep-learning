import os
import concurrent.futures
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin, urlparse
import requests, zipfile, io


def is_valid(url):
    """
    Проверяем, является ли url действительным URL
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def download(url, file_name):
    """
    Загружает файл по URL‑адресу и помещает его в папку pathname
    """
    response = requests.get(url)
    try:
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(file_name)
    except zipfile.BadZipfile:
        return


def get_all_images(url):
    """
    Возвращает все URL‑адреса изображений по url одной странице
    """
    soup = bs(requests.get(url).content, "html.parser")
    names = []
    url_names = []
    for img in soup.find_all("img"):
        img_url = img.attrs.get("src")
        if not img_url:
            continue
        img_url = urljoin(url, img_url)
        try:
            pos = img_url.index("?")
            img_url = img_url[:pos]
        except ValueError:
            pass
        if is_valid(img_url):
            file_name = os.path.join(img_url, img_url.split("/")[-1])
            file_name = os.path.splitext(file_name)[0] + ".zip"
            url_img = "https://publicdomainvectors.org/download.php?file=" + os.path.basename(file_name)
            names.append("./ai-pictures/" + os.path.basename(file_name))
            url_names.append(url_img)

    return url_names, names


def get_pics_one_page(category):
    """
    скачиваем картинки с одной страницы одной категории
    """
    img_urles, names = get_all_images(category)
    for url, name in zip(img_urles, names):
        download(url, name)


dict_urls = {
    "https://publicdomainvectors.org/en/free-clipart/animals/date/ai/360/": 6,
    "https://publicdomainvectors.org/en/free-clipart/architecture/date/ai/360/": 2,
    "https://publicdomainvectors.org/en/free-clipart/backgrounds/date/ai/360/": 10,
    "https://publicdomainvectors.org/en/free-clipart/business/date/ai/360/": 1,
    "https://publicdomainvectors.org/en/free-clipart/food-and-drink/date/ai/360/": 3,
    "https://publicdomainvectors.org/en/free-clipart/nature/date/ai/360/": 5,
    "https://publicdomainvectors.org/en/free-clipart/objects/date/ai/360/": 28,
    "https://publicdomainvectors.org/en/free-clipart/people/date/ai/360/": 15,
    "https://publicdomainvectors.org/en/free-clipart/transportation/date/ai/360/": 3
}

with concurrent.futures.ThreadPoolExecutor() as executor:
    for (site, number_page) in dict_urls.items():
        print(f"URL of site:{site}")
        tasks = [executor.submit(get_pics_one_page, site + str(i)) for i in range(1, number_page + 1)]
        results = [task.result() for task in concurrent.futures.as_completed(tasks)]
