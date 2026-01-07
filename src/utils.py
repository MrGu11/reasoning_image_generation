import os
import random
import cv2
import numpy as np


def ensure_dir(path: str):
    """确保目录存在"""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def rand_color(min_v=30, max_v=220):
    """生成随机 BGR 颜色"""
    return tuple(int(random.uniform(min_v, max_v)) for _ in range(3))


def save_image(im: np.ndarray, path: str):
    """保存图像 (cv2 格式: numpy.ndarray, BGR)"""
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, im)

