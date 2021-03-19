import cv2
from PIL import Image
import numpy as np


def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def open_jpg(path):
    return np.array(Image.open(path))


def open_png(path):
    return np.array(Image.open(path))


def split_left_right(image):
    w = image.shape[1]

    w = w // 2
    left = image[:, :w, :]
    right = image[:, w:, :]
    return left, right
