import unittest

import numpy as np
from numpy.linalg import norm

from try_gan.tests.fixture import Fixture
from try_gan import perturb
from try_gan import normalize


class TestPerturb(unittest.TestCase):
    def setUp(self):
        f = Fixture()
        self.image = f.make_image()
        self.normalized = f.make_normalized_image()

    def tearDown(self):
        pass

    def get_r(self):
        return np.random.RandomState(42)

    def test_gauss(self):
        actual = perturb.gauss_noise(self.normalized, self.get_r(), mu=0.0, sigma=0.2)
        r = self.get_r()
        noise = r.normal(0.0, 0.2, size=(8, 8, 3))
        expected = Fixture().make_normalized_image() + noise
        self.assertAlmostEqual(0, norm(actual - expected))

    def test_salt_and_pepper(self):
        actual = self.normalized
        perturb.salt_and_pepper(actual, self.get_r(), s_vs_p=0.25, amount=0.2)

        expected = np.array(
            [
                [
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.85098039, 0.85098039, 0.85098039],
                    [0.03529412, 0.03529412, 0.03529412],
                    [0.26666667, 0.26666667, 0.26666667],
                    [0.0, 0.70588235, 0.70588235],
                    [1.0, 1.0, 1.0],
                    [0.99215686, 0.99215686, 0.99215686],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.70196078, 0.70196078, 0.70196078],
                    [0.00784314, 0.00784314, 0.00784314],
                    [0.00392157, 0.0, 0.00392157],
                    [0.00784314, 0.00784314, 0.00784314],
                    [0.21176471, 0.21176471, 0.21176471],
                    [0.84705882, 0.84705882, 0.84705882],
                ],
                [
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.98823529, 0.98823529, 0.98823529],
                    [0.21176471, 0.0, 0.21176471],
                    [0.00784314, 0.0, 0.00784314],
                    [0.0, 0.0, 0.0],
                    [0.01568627, 0.01568627, 0.01568627],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.97647059, 1.0, 0.97647059],
                    [0.0, 0.0, 0.83529412],
                    [0.04313725, 0.04313725, 0.04313725],
                    [0.0, 0.00784314, 0.00784314],
                    [0.0, 0.0, 0.0],
                    [0.00392157, 0.00392157, 0.00392157],
                ],
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.57254902, 0.57254902, 0.57254902],
                    [1.0, 0.0, 0.01960784],
                    [0.00784314, 0.00784314, 0.00784314],
                    [0.00784314, 0.00784314, 0.00784314],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.98431373, 0.98431373, 0.98431373],
                    [0.57254902, 0.57254902, 0.57254902],
                    [0.0, 0.00392157, 0.00392157],
                    [0.00392157, 0.00392157, 0.00392157],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.56862745],
                    [0.00392157, 0.00392157, 0.00392157],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.99215686, 0.99215686, 0.99215686],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.77254902, 0.77254902, 0.77254902],
                ],
            ]
        )
        self.assertAlmostEqual(0, norm(actual - expected))

    def test_normalizer(self):
        p = perturb.Normalizer()
        actual = p.perturb(self.image)
        expected = self.normalized
        self.assertAlmostEqual(0, norm(expected - actual))

        expected = self.image
        actual = Fixture().make_image()
        self.assertAlmostEqual(0, norm(expected - actual))

    def test_discetizer(self):
        p = perturb.Discretizer()
        actual = p.perturb(self.normalized)
        expected = self.image
        self.assertAlmostEqual(0, norm(expected - actual))

        expected = self.normalized
        actual = Fixture().make_normalized_image()
        self.assertAlmostEqual(0, norm(expected - actual))

    def test_gauss_perturber(self):
        sigma = 0.1
        p = perturb.GaussPerturber(self.get_r(), sigma=sigma)
        actual = p.perturb(self.normalized)

        expected = perturb.gauss_noise(self.normalized, self.get_r(), mu=0.0, sigma=sigma)
        self.assertAlmostEqual(0, norm(actual - expected))

    def test_snp_perturber(self):
        s_vs_p = 0.3
        amount = 0.1
        p = perturb.SNPPerturber(self.get_r(), s_vs_p=s_vs_p, amount=amount)
        actual = p.perturb(self.normalized)
        expected = Fixture().make_normalized_image()
        perturb.salt_and_pepper(expected, self.get_r(), s_vs_p=s_vs_p, amount=amount)
        self.assertAlmostEqual(0, norm(actual - expected))

    def test_composite_perturber(self):
        sigma = 0.1
        r = self.get_r()
        g = perturb.GaussPerturber(r, sigma=sigma)
        s_vs_p = 0.3
        amount = 0.1
        snp = perturb.SNPPerturber(r, s_vs_p=s_vs_p, amount=amount)
        ops = [perturb.Normalizer(), g, snp, perturb.Discretizer()]
        p = perturb.CompositePerturber(ops)
        actual = p.perturb(self.image)

        f = Fixture()
        r = self.get_r()
        image = f.make_normalized_image()
        image = perturb.gauss_noise(image, r=r, sigma=sigma)
        perturb.salt_and_pepper(image, r=r, s_vs_p=s_vs_p, amount=amount)
        expected = normalize.normal_to_bytes(image)
        self.assertAlmostEqual(0, norm(actual - expected))
