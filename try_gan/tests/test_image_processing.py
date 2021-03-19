import unittest

from numpy.linalg import norm
import numpy as np
import cv2
import os

from try_gan.tests.fixture import Fixture
from try_gan import image_files


class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        f = Fixture()
        self.zero, self.one, self.two, self.three = f.make_test_digits()

    def tearDown(self):
        pass

    def test_open_png(self):
        f = Fixture()

        path = os.path.join(f.all_four, "zero.png")
        expected = self.zero
        actual = image_files.open_png(path)
        self.assertAlmostEqual(0, norm(expected - actual))

        path = os.path.join(f.all_four, "one.png")
        expected = self.one
        actual = image_files.open_png(path)
        self.assertAlmostEqual(0, norm(expected - actual))

        path = os.path.join(f.all_four, "two.png")
        expected = self.two
        actual = image_files.open_png(path)
        self.assertAlmostEqual(0, norm(expected - actual))

        path = os.path.join(f.all_four, "three.png")
        expected = self.three
        actual = image_files.open_png(path)
        self.assertAlmostEqual(0, norm(expected - actual))

    def test_bgr2rgb(self):
        f = Fixture()

        path = os.path.join(f.all_four, "one.png")
        expected = image_files.open_png(path)
        # by default opencv reads images as BGR
        bgr_image = cv2.imread(path)
        actual = image_files.bgr2rgb(bgr_image)
        self.assertAlmostEqual(0, norm(expected - actual))

    def test_split_image(self):
        f = Fixture()

        path = os.path.join(f.concatenated, "zero_one.png")
        zero_one = image_files.open_png(path)
        left, right = image_files.split_left_right(zero_one)

        expected = self.zero
        actual = left
        self.assertAlmostEqual(0, norm(expected - actual))

        expected = self.one
        actual = right
        self.assertAlmostEqual(0, norm(expected - actual))

        path = os.path.join(f.concatenated, "two_three.png")
        two_three = image_files.open_png(path)
        left, right = image_files.split_left_right(two_three)

        expected = self.two
        actual = left
        self.assertAlmostEqual(0, norm(expected - actual))

        expected = self.three
        actual = right
        self.assertAlmostEqual(0, norm(expected - actual))
