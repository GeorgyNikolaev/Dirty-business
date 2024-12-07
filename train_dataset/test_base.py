import os
import sys
import json
import base64
import cv2
import numpy as np
import io

def convert_mask_to_yolo(mask_image):
    # Открываем изображение
    # Поскольку теперь мы работаем напрямую с загруженным из сабмита изображением,
    # mask_image уже является numpy массивом.

    if mask_image is None:
        print("Не удалось обработать маску")
        return None

    # Создаем маску для черного цвета
    black_mask = cv2.inRange(mask_image, (0, 0, 0), (50, 50, 50))

    # Создаем новое изображение, где черный цвет остается, а остальные цвета становятся белыми
    new_image = np.ones_like(mask_image) * 255  # Начинаем с белого изображения
    new_image[black_mask > 0] = [0, 0, 0]  # Заменяем черные пиксели

    # Преобразуем в градации серого для нахождения контуров
    return new_image


# Расчет IoU через векторы
def iou(y_true, y_pred, class_label):
    y_true = y_true == class_label
    y_pred = y_pred == class_label
    if y_true.sum() == 0 and y_pred.sum() == 0:
        return 1.0
    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    return inter / (union + 1e-8)


def batch_iou(submit_bytestream, gt_path):
    # Теперь мы не читаем маски сабмита из директории, а загружаем их из файла submit
    # submit_bytestream - bytestream JSON файла, созданному в infer_by_test.py
    
    submit_dict = json.load(io.BytesIO(f))

    res = []
    gt_masks_list = {fn for fn in os.listdir(gt_path) if fn.lower().endswith(".png") or fn.lower().endswith(".jpg")}
    for image_name, base64_mask_str in submit_dict.items():
        # Предполагаем, что GT маски имеют то же имя, что и исходное изображение, но с расширением .png
        gt_mask_name = image_name.replace(".jpg", ".png")
        if gt_mask_name in gt_masks_list:
            # Декодируем баз64 в байты
            sb_mask_bytes = base64.b64decode(base64_mask_str)
            # Декодируем изображение из памяти
            sb_mask_img = cv2.imdecode(np.frombuffer(sb_mask_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Загружаем gt маску с диска
            gt_mask_path = os.path.join(gt_path, gt_mask_name)
            gt_mask_img = cv2.imread(gt_mask_path)

            # Применяем convert_mask_to_yolo к обоим маскам
            gt_mask_yolo = convert_mask_to_yolo(gt_mask_img)
            sb_mask_yolo = convert_mask_to_yolo(sb_mask_img)

            if gt_mask_yolo is not None and sb_mask_yolo is not None:
                res.append(iou(gt_mask_yolo, sb_mask_yolo, 255))
    return res


if __name__ == '__main__':
    submit_bytestream, gt_path = sys.argv[1:]
    res = np.mean(batch_iou(submit_bytestream, gt_path))
    print(f"{abs(res):.4f}")
