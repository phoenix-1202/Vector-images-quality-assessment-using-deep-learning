import re
from tqdm import tqdm
import os
import multiprocessing
# from cairosvg import svg2png
import xml.etree.ElementTree


def rename_files(input_directory):
    for filename in os.listdir(input_directory):
        new_name = re.sub('[\W_]', '', filename) + '.svg'
        os.rename(os.path.join(input_directory, filename),
                  os.path.join(input_directory, new_name))


def to_png(svg_file):
    output_directory = './data/png-pictures/'
    output_path = svg_file.replace('.svg', '.png')
    output_path = os.path.basename(output_path)
    png_file = os.path.join(output_directory, output_path)
    if not os.path.exists(png_file):
        try:
            # svg2png(url=svg_file, write_to=png_file)
            pass
        except xml.etree.ElementTree.ParseError:
            return


def process_files_in_parallel(files):
    num_processes = 60
    pool = multiprocessing.Pool(processes=num_processes)
    pool.map(to_png, files)
    pool.close()
    pool.join()


def convert_to_png(input_directory):
    svg_files = [os.path.join(input_directory, f) for f in
                 tqdm(os.listdir(input_directory), desc="Конвертируем картинки в png формат") if
                 f.endswith('.svg')]
    process_files_in_parallel(svg_files)
