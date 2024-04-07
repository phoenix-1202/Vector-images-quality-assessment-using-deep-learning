import os
from PIL import Image
import re
from cairosvg import svg2png
from tqdm import tqdm

def rename_files(input_directory):
    i = 0
    for filename in os.listdir(input_directory):
        new_name = re.sub('[\W_]', '', filename) + '.svg'  # Удаляем все символы, кроме букв и цифр
        os.rename(os.path.join(input_directory, filename), os.path.join(input_directory, new_name))  # Переименовываем файл
        #i += 1
        #if os.path.isfile(os.path.join(input_directory, filename)):
            #name, ext = os.path.splitext(filename)
            #new_filename = f"image{i}" + ext
            #os.rename(os.path.join(input_directory, filename), os.path.join(input_directory, new_filename))


def convert_to_png(input_directory, output_directory):
    for file in tqdm(os.listdir(input_directory), desc="Конвертируем картинки в png формат"):
        input_path = os.path.join(input_directory, file)
        filename = os.path.splitext(file)[0] + ".png"
        output_path = os.path.join(output_directory, filename)
        if input_path.endswith(".svg"):
            #svg2png(url=input_path, write_to=output_path)
            try:
                svg2png(url=input_path, write_to=output_path)
            except Exception:
                continue
        else:
            try:
                input_image = Image.open(input_path)
                input_image.save(output_path, 'PNG')
            except Exception:
                continue
