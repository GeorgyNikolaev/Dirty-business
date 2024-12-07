import os
import cv2
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

target_size_img = (256, 144)

TRUE_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
IMAGES_DIR_TRUE = TRUE_PATH + '/train_dataset/cv_open_dataset/open_img'  # Путь к вашему датасету с изображениями
IMAGES_DIR_SINT = TRUE_PATH + '/train_dataset/cv_synt_dataset/synt_img'  # Путь к вашему датасету с изображениями
MASKS_DIR_TRUE = TRUE_PATH + '/train_dataset/cv_open_dataset/open_msk'  # Путь к вашему датасету с масками
MASKS_DIR_SINT = TRUE_PATH + '/train_dataset/cv_synt_dataset/synt_msk'  # Путь к вашему датасету с масками

image_files_true = [f for f in os.listdir(IMAGES_DIR_TRUE) if f.endswith(('.jpg', '.png'))]
image_files_synt = [f for f in os.listdir(IMAGES_DIR_SINT) if f.endswith(('.jpg', '.png'))]
mask_files_true = [f for f in os.listdir(MASKS_DIR_TRUE) if f.endswith('.png')]
mask_files_synt = [f for f in os.listdir(MASKS_DIR_SINT) if f.endswith('.png')]


def read() -> (np.array, np.array):
    x_data = []
    for image_path in image_files_true:
        # Читаем и резайзим изображение, чтобы все были одинаковые
        image = cv2.imread(IMAGES_DIR_TRUE + '/' + image_path)
        image = cv2.resize(image, target_size_img)
        image = image / 255
        x_data.append(image)

    for image_path in image_files_synt:
        image = cv2.imread(IMAGES_DIR_SINT + '/' + image_path)
        image = cv2.resize(image, target_size_img)
        image = image / 255
        x_data.append(image)

    y_data = []
    for mask_path in mask_files_true:
        # Читаем, резайзим изображение, чтобы все были одинаковые и приводим к бинарному виду
        mask = cv2.imread(MASKS_DIR_TRUE + '/' + mask_path)
        mask = cv2.resize(mask, target_size_img)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask / 255
        binary_mask.reshape((len(binary_mask), len(binary_mask[0]), 1))
        y_data.append(binary_mask)

    for mask_path in mask_files_synt:
        mask = cv2.imread(MASKS_DIR_SINT + '/' + mask_path)
        mask = cv2.resize(mask, target_size_img)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask / 255
        binary_mask.reshape((len(binary_mask), len(binary_mask[0]), 1))
        y_data.append(binary_mask)

    return (np.array(x_data, dtype=float), np.array(y_data, dtype=float))


def get_metric(data):
    pass

