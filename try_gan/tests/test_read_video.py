import unittest

import numpy as np

from try_gan import read_video


class TestReadVideo(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_generator(self, n=10):
        def generator():
            for i in range(n):
                yield i * i
        return generator

    def test_example_generator(self):
        expected = [i * i for i in range(5)]
        actual = list(self.make_generator(n=5)())
        self.assertEqual(expected, actual)

    def test_sample_from_generator(self):
        g = self.make_generator()
        h = read_video.sample_from_generator(g(), 10, 3, np.random.RandomState(42))
        actual = list(h)
        expected = [1, 25, 64]
        self.assertEqual(expected, actual)

        g = self.make_generator()
        h = read_video.sample_from_generator(g(), 10, 3, np.random.RandomState(48))
        actual = list(h)
        expected =  [4, 25, 49]
        self.assertEqual(expected, actual)

        g = self.make_generator()
        h = read_video.sample_from_generator(g(), 10, 3, np.random.RandomState(51))
        actual = list(h)
        expected = [4, 9, 16]
        self.assertEqual(expected, actual)
