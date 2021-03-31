import numpy as np
from PIL import Image
import os

from try_gan import read_video
from try_gan import perturb
from try_gan.pipeline_from_files import JpgFilePipeline as Pipeline


class Framer:
    def __init__(self, frames: read_video.Frames, r=None, p=None):
        self.frames = frames
        if r is None:
            r = np.random.RandomState(42)
        self.r = r

        if p is None:
            sigma = 0.1
            g = perturb.GaussPerturber(r, sigma=sigma)
            s_vs_p = 0.3
            amount = 0.1
            snp = perturb.SNPPerturber(r, s_vs_p=s_vs_p, amount=amount)
            ops = [perturb.Normalizer(), g, snp, perturb.Discretizer()]
            p = perturb.CompositePerturber(ops)
        self.p = p

    def write_samples(self, path, n):
        sampler = read_video.sample_from_generator(self.frames, frame_count, n, self.r)
        for i, frame in enumerate(read_video.concatenated_frames(sampler, self.p)):
            filename = os.path.join(path, "{}.jpg".format(i))
            image = Image.fromarray(frame)
            image.save(filename)
        self.cleaunup()


class FramerPipeline(Pipeline):
    def __init__(
        self,
        path,
        framer,
        test_framer=None,
        train_sample_count=400,
        test_sample_count=40,
    ):
        Pipeline.__init__(self, path)
        self.framer = framer
        if test_framer is None:
            test_framer = framer
        self.test_framer = test_framer
        self.train_sample_count = train_sample_count
        self.test_sample_count = test_sample_count

    def _train_dir(self):
        return os.path.join(self.PATH, "train")

    def _test_dir(self):
        return os.path.join(self.PATH, "test")

    @property
    def train_dir(self):
        return self._train_dir()

    @property
    def test_dir(self):
        return self._test_dir()

    def make_train(self):
        self.framer.write_samples(self.train_dir, self.train_sample_count)
        return Pipeline.make_train(self)

    def make_test(self):
        self.test_framer.write_samples(self.test_dir, self.test_sample_count)
        return Pipeline.make_test(self)
