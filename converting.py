import os
from PIL import Image


def rename_files(input_directory):
    i = 0
    for filename in os.listdir(input_directory):
        i += 1
        if os.path.isfile(os.path.join(input_directory, filename)):
            name, ext = os.path.splitext(filename)
            new_filename = f"image{i}" + ext
            os.rename(os.path.join(input_directory, filename), os.path.join(input_directory, new_filename))


def convert_to_png(input_directory, output_directory):
    for file in os.listdir(input_directory):
        input_path = os.path.join(input_directory, file)
        filename = os.path.splitext(file)[0] + ".png"
        output_path = os.path.join(output_directory, filename)
        if input_path.endswith(".svg"):
            try:
                continue
                # svg2png(url=input_path, write_to=output_path)
                # todo:: поймать норм ошибку
            except Exception:
                continue
        else:
            try:
                input_image = Image.open(input_path)
                input_image.save(output_path, 'PNG')
            except Exception:
                continue
