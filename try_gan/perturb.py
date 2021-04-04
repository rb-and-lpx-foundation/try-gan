import cv2
import numpy as np

from try_gan import normalize
from try_gan.image_files import make_square, rgb2gray


def gauss_noise(image, r: np.random.RandomState, mu=0.0, sigma=0.05):
    noise = r.normal(mu, sigma, size=image.shape)
    return image + noise


def salt_and_pepper(image, r: np.random.RandomState, s_vs_p=0.5, amount=0.004):
    # Salt sets pixels to all the way on for a channel
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [r.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image[tuple(coords)] = 1

    # Pepper sets pixels to all the way off for a channel
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [r.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image[tuple(coords)] = 0


def random_color(r):
    def channel():
        if r.random() > 0.5:
            return 255
        else:
            return 0

    return [channel() for _ in range(3)]


def random_coord(w, h, r):
    x = r.randint(w)
    y = r.randint(h)
    return x, y


def random_rect(w, h, radius, r):
    x, y = random_coord(w, h, r)
    u = r.randint(radius // 2) + 1
    v = r.randint(radius // 2) + 1
    p0 = (x - u, y - v)
    p1 = (x + u, y + v)
    return p0, p1


def random_thickeness(d, r):
    if r.random() < 0.5:
        return -1
    return r.randint(d) + 1


class Perturber:
    def perturb(self, image):
        return image


class Normalizer(Perturber):
    def perturb(self, image):
        return normalize.normalize_to_floats(image)


class Discretizer(Perturber):
    def perturb(self, image):
        normalize.clip_floats(image)
        return normalize.normal_to_bytes(image)


class GaussPerturber(Perturber):
    def __init__(self, r: np.random.RandomState, mu=0.0, sigma=0.05):
        self.r = r
        self.mu = mu
        self.sigma = sigma

    def perturb(self, image):
        return gauss_noise(image, self.r, mu=self.mu, sigma=self.sigma)


class SNPPerturber(Perturber):
    def __init__(self, r: np.random.RandomState, s_vs_p=0.5, amount=0.004):
        self.r = r
        self.s_vs_p = s_vs_p
        self.amount = amount

    def perturb(self, image):
        salt_and_pepper(image, self.r, s_vs_p=self.s_vs_p, amount=self.amount)
        return image


class CompositePerturber(Perturber):
    def __init__(self, ops):
        self.ops = ops

    def perturb(self, image):
        for op in self.ops:
            image = op.perturb(image)
        return image


class SquarePerturber(Perturber):
    def __init__(self, r: np.random.RandomState, max_count, thickness, radius):
        self.r = r
        self.max_count = max_count
        self.thickenss = thickness
        self.radius = radius

    def perturb(self, image):
        w = image.shape[0]
        h = image.shape[1]
        for _ in range(self.r.randint(self.max_count)):
            color = random_color(self.r)
            thickness = random_thickeness(self.thickenss, self.r)
            p, q = random_rect(w, h, self.radius, self.r)
            cv2.rectangle(image, p, q, color=color, thickness=thickness)
        return image


class ToSquarePerturber(Perturber):
    def __init__(self, dim):
        self.dim = dim

    def perturb(self, image):
        return make_square(image, self.dim)


class ConcatenatePerturber(Perturber):
    def __init__(self, perturber: Perturber):
        self.perturber = perturber

    def perturb(self, image):
        perturbed = self.perturber.perturb(image)
        return np.concatenate([image, perturbed], axis=1)


class MaybePerturber(Perturber):
    def __init__(self, perturber: Perturber, prob, r: np.random.RandomState):
        self.perturber = perturber
        self.prob = prob
        self.r = r

    def perturb(self, image):
        if self.r.random() > self.prob:
            return image
        else:
            return self.perturber.perturb(image)


class BlackoutPerturber(Perturber):
    def perturb(self, image):
        return np.zeros_like(image)


class IdentityPerturber(Perturber):
    def perturb(self, image):
        return image


class GrayPerturber(Perturber):
    def perturb(self, image):
        return rgb2gray(image)
