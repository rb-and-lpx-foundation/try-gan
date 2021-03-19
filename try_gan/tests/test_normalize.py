import unittest

from numpy.linalg import norm
import numpy as np

from try_gan import normalize


class TestNormalize(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_clip(self):
        actual = np.array([[0, 1, 1], [-0.1, 0.5, 1.2]])
        normalize.clip_floats(actual)
        expected = np.array([[0.0, 1.0, 1.0], [0, 0.5, 1.0]])
        self.assertAlmostEqual(0, norm(actual - expected))

    def test_normalize_to_floats(self):
        x = np.array([0, 128, 255])
        y = normalize.normalize_to_floats(x)

        # the original does not change
        actual = x
        expected = np.array([0, 128, 255])
        self.assertAlmostEqual(0, norm(actual - expected))

        # the new one is a normalized copy
        actual = y
        expected = np.array([0, 128.0 / 255.0, 1.0])
        self.assertAlmostEqual(0, norm(actual - expected))

    def test_normal_to_uint8(self):
        x = np.array([0, 128.0 / 255.0, 1.0])
        y = normalize.normal_to_bytes(x)

        # the original does not change
        actual = x
        expected = np.array([0, 128.0 / 255.0, 1.0])
        self.assertAlmostEqual(0, norm(actual - expected))

        # the new one is a normalized copy
        actual = y
        expected = np.array([0, 128, 255])
        self.assertAlmostEqual(0, norm(actual - expected))
