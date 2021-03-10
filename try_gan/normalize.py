import numpy as np


def clip_floats(image, lower_bound=0.0, upper_bound=1.0):
    image[image < lower_bound] = lower_bound
    image[image > upper_bound] = upper_bound


def normalize_to_floats(image):
    return image.copy() / 255.0


def normal_to_bytes(image):
    return (image.copy() * 255.5).astype(np.uint8)
