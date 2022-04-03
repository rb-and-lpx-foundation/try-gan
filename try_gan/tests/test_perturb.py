import unittest

import numpy as np
from numpy.linalg import norm

from try_gan.tests.fixture import Fixture
from try_gan import perturb
from try_gan import normalize
from try_gan.image_files import split_left_right


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

    def test_discretizer(self):
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

        expected = perturb.gauss_noise(
            self.normalized, self.get_r(), mu=0.0, sigma=sigma
        )
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
        normalize.clip_floats(image)
        expected = normalize.normal_to_bytes(image)
        self.assertAlmostEqual(0, norm(actual - expected))

    def test_concatenate_perturber(self):
        image = self.image.copy()
        # concatenate image with copy of image
        copy_copy = perturb.ConcatenatePerturber(perturb.IdentityPerturber())
        perturbed = copy_copy.perturb(image)
        left, right = split_left_right(perturbed)
        self.assertAlmostEqual(0, norm(left - self.image))
        self.assertAlmostEqual(0, norm(right - self.image))

        # concatenate image with blacked out image
        copy_black = perturb.ConcatenatePerturber(perturb.BlackoutPerturber())
        perturbed = copy_black.perturb(image)
        left, right = split_left_right(perturbed)
        self.assertAlmostEqual(0, norm(left - self.image))
        self.assertAlmostEqual(0, norm(right))

    def test_maybe_perturber(self):
        r = np.random.RandomState(42)
        make_black = [r.random() < 0.75 for _ in range(4)]
        self.assertTrue(make_black[0])
        self.assertFalse(make_black[1])
        self.assertTrue(make_black[2])
        self.assertTrue(make_black[3])

        # make a perturber which will make image black 3 out of four times
        image = self.image.copy()
        p = perturb.MaybePerturber(
            perturber=perturb.BlackoutPerturber(),
            prob=0.75,
            r=np.random.RandomState(42),
        )
        self.assertAlmostEqual(0, norm(p.perturb(image)))
        self.assertAlmostEqual(0, norm(p.perturb(image) - self.image))
        self.assertAlmostEqual(0, norm(p.perturb(image)))
        self.assertAlmostEqual(0, norm(p.perturb(image)))

    def test_to_square_perturber(self):
        p = perturb.ToSquarePerturber(4)
        image = self.image[2:4, 2:6, :]
        self.assertEqual((2, 4, 3), image.shape)
        actual = p.perturb(image)
        self.assertEqual((4, 4, 3), actual.shape)
        expected = np.array(
            [
                [[249, 249, 249], [213, 213, 213], [11, 11, 11], [2, 2, 2]],
                [[252, 252, 252], [54, 54, 54], [2, 2, 2], [0, 0, 0]],
                [[249, 249, 249], [213, 213, 213], [11, 11, 11], [2, 2, 2]],
                [[252, 252, 252], [54, 54, 54], [2, 2, 2], [0, 0, 0]],
            ],
            dtype=np.uint8,
        )
        self.assertAlmostEqual(0, norm(actual - expected))

    def test_blackout_perturber(self):
        image = self.image.copy()
        p = perturb.BlackoutPerturber()
        black = p.perturb(image)

        # want the original image unaltered
        expected = self.image
        actual = image
        self.assertAlmostEqual(0, norm(actual - expected))

        # want black image to have same shape and type as original, but only zero entries
        self.assertEqual(image.shape, black.shape)
        self.assertEqual(image.dtype, black.dtype)
        self.assertAlmostEqual(0, norm(black))

    def test_identity_perturber(self):
        image = self.image.copy()
        p = perturb.IdentityPerturber()
        expected = self.image
        actual = p.perturb(image)
        self.assertAlmostEqual(0, norm(actual - expected))
