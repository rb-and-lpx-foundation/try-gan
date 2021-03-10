import numpy as np
from try_gan import normalize


def noisy(noise_typ, image, r: np.random.RandomState):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = r.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [r.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [r.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = r.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = r.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


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
