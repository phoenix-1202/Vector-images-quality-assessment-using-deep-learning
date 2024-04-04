import cv2
import os
from visualization import visualize_distribution


def get_distribution(input_directory, output_directory):
    list_width = []
    list_height = []
    list_error = []
    for filename in os.listdir(input_directory):
        path_file = os.path.join(input_directory, filename)
        try:
            image = cv2.imread(path_file)
            width, height, _ = image.shape
            list_width.append(width)
            list_height.append(height)
        except Exception as someReadingError:
            list_error.append(path_file)
            continue

    visualize_distribution(list_width, list_height, output_directory)


def normalize_images(input_directory, output_directory):
    list_width = []
    list_height = []
    for filename in os.listdir(input_directory):
        path_file = os.path.join(input_directory, filename)
        input_image = cv2.imread(path_file)

        if input_image is None:
            os.remove(path_file)
            continue

        width, height, _ = input_image.shape
        if width > 7000 or height > 7000:
            width, height = 7000, 7000
            resized_image = cv2.resize(input_image, (width, height), interpolation=cv2.INTER_CUBIC)
            os.remove(path_file)
            cv2.imwrite(path_file, resized_image)
        list_width.append(width)
        list_height.append(height)

    visualize_distribution(list_width, list_height, output_directory)
    return list_width, list_height
