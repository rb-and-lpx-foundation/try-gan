import unittest

import os
import numpy as np
from numpy.linalg import norm

from try_gan import perturb
from try_gan import image_files
from try_gan import read_video
from try_gan.tests.fixture import Fixture


class FramesFromFixture(read_video.Frames):
    def __init__(self):
        f = Fixture()
        frames = f.make_test_digits()
        read_video.Frames.__init__(self, frames, 4, 8.0)
        self.has_cleanedup = False

    def cleanup(self):
        self.has_cleanedup = True


class TestReadVideo(unittest.TestCase):
    def setUp(self):
        self.test_frames = FramesFromFixture()

    def tearDown(self):
        pass

    def make_generator(self, n=10):
        def generator():
            for i in range(n):
                yield i * i

        return generator

    def make_perturber(self, r):
        sigma = 0.1
        g = perturb.GaussPerturber(r, sigma=sigma)
        s_vs_p = 0.3
        amount = 0.1
        snp = perturb.SNPPerturber(r, s_vs_p=s_vs_p, amount=amount)
        ops = [perturb.Normalizer(), g, snp, perturb.Discretizer()]
        return perturb.CompositePerturber(ops)

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
        expected = [4, 25, 49]
        self.assertEqual(expected, actual)

        g = self.make_generator()
        h = read_video.sample_from_generator(g(), 10, 3, np.random.RandomState(51))
        actual = list(h)
        expected = [4, 9, 16]
        self.assertEqual(expected, actual)

        # try asking for more samples than available
        g = self.make_generator(n=3)
        h = read_video.sample_from_generator(g(), 3, 300, np.random.RandomState(42))
        actual = list(h)
        expected = [0, 1, 4]
        self.assertEqual(expected, actual)

    def test_frames_test_object(self):
        f = Fixture()
        expected_frames = f.make_test_digits()

        self.assertFalse(self.test_frames.has_cleanedup)
        self.assertEqual(4, self.test_frames.frame_count)
        self.assertEqual(8.0, self.test_frames.fps)

        for i, actual in enumerate(self.test_frames.frames):
            expected = expected_frames[i]
            self.assertAlmostEqual(0, norm(expected - actual))

        self.test_frames.cleanup()
        self.assertTrue(self.test_frames.has_cleanedup)

    def test_perturbed_frames(self):
        f = Fixture()
        p0 = self.make_perturber(np.random.RandomState(42))
        p1 = self.make_perturber(np.random.RandomState(42))
        expected_frames = [p0.perturb(digit) for digit in f.make_test_digits()]
        perturbed_frames = read_video.PerturbedFrames(self.test_frames, p1)
        self.assertEqual(4, perturbed_frames.frame_count)
        self.assertEqual(8.0, perturbed_frames.fps)
        for i, actual in enumerate(perturbed_frames.frames):
            expected = expected_frames[i]
            self.assertAlmostEqual(0, norm(expected - actual))

        self.assertFalse(self.test_frames.has_cleanedup)
        perturbed_frames.cleanup()
        self.assertTrue(self.test_frames.has_cleanedup)

    def test_sampled_frames(self):
        r = np.random.RandomState(43)
        indices = r.choice(4, size=2, replace=False)
        expected = [2, 1]
        actual = list(indices)
        self.assertEqual(expected, actual)

        r = np.random.RandomState(43)
        sampled_frames = read_video.SampledFrames(self.test_frames, 2, r)
        self.assertEqual(4, sampled_frames.frame_count)
        self.assertEqual(8.0, sampled_frames.fps)

        digits = Fixture().make_test_digits()
        actual = list(sampled_frames.frames)
        self.assertEqual(2, len(actual))
        self.assertAlmostEqual(0, norm(digits[1] - actual[0]))
        self.assertAlmostEqual(0, norm(digits[2] - actual[1]))

        self.assertFalse(self.test_frames.has_cleanedup)
        sampled_frames.cleanup()
        self.assertTrue(self.test_frames.has_cleanedup)

    def test_frames_from_glob(self):
        f = Fixture()
        digits = f.make_test_digits()
        g = os.path.join(f.all_four, "*.png")
        glob_frames = read_video.GlobFrames(
            g, 35.0, open_image_file=image_files.open_png
        )

        self.assertEqual(4, glob_frames.frame_count)
        self.assertEqual(35.0, glob_frames.fps)

        actual = list(glob_frames.frames)
        self.assertEqual(4, len(actual))
        self.assertAlmostEqual(0, norm(digits[1] - actual[0]))
        self.assertAlmostEqual(0, norm(digits[3] - actual[1]))
        self.assertAlmostEqual(0, norm(digits[2] - actual[2]))
        self.assertAlmostEqual(0, norm(digits[0] - actual[3]))
