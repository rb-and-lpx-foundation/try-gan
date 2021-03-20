#!/usr/bin/env python
import unittest

from try_gan.tests.test_example import TestExample
from try_gan.tests.test_image_processing import TestImageProcessing
from try_gan.tests.test_normalize import TestNormalize
from try_gan.tests.test_perturb import TestPerturb
from try_gan.tests.test_read_video import TestReadVideo


class CountSuite(object):
    def __init__(self):
        self.count = 0
        self.s = unittest.TestSuite()

    def add(self, tests):
        self.count += 1
        print("%d: %s" % (self.count, tests.__name__))
        self.s.addTest(unittest.makeSuite(tests))


def suite():
    s = CountSuite()

    s.add(TestExample)
    s.add(TestImageProcessing)
    s.add(TestNormalize)
    s.add(TestPerturb)
    s.add(TestReadVideo)

    return s.s


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
