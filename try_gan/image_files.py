import cv2
from PIL import Image
import numpy as np


def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def rgb2lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def lab2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_LAB2LRGB)


def rgb2gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def open_jpg(path):
    return np.array(Image.open(path))


def save_image(image, path):
    image = Image.fromarray(image)
    image.save(path)


def open_png(path):
    return np.array(Image.open(path))


def split_left_right(image):
    w = image.shape[1]

    w = w // 2
    left = image[:, :w, :]
    right = image[:, w:, :]
    return left, right


def compute_padding(total, current):
    leading = (total - current) // 2
    following = total - (current + leading)
    return leading, following


def compute_square_padding(m, n):
    total = max(m, n)
    left, right = compute_padding(total, m)
    top, bottom = compute_padding(total, n)
    return left, right, top, bottom


def make_square(image, dim):
    m = image.shape[0]
    n = image.shape[1]
    left, right, top, bottom = compute_square_padding(m, n)
    image = cv2.copyMakeBorder(image, left, right, top, bottom, cv2.BORDER_WRAP)
    return cv2.resize(image, dsize=(dim, dim))
