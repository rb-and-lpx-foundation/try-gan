from PIL import Image
import os

from try_gan import read_video
from try_gan import perturb
from try_gan.pipeline_from_files import JpgFilePipeline as Pipeline


class Framer:
    def __init__(self, r, p):
        self.r = r
        self.p = perturb.ConcatenatePerturber(p)

    def fetch_frames(self):
        raise NotImplementedError()

    def write_samples(self, path, n):
        sampler = read_video.SampledFrames(self.fetch_frames(), n, self.r)
        perturbed = read_video.PerturbedFrames(sampler, self.p)
        for i, frame in enumerate(perturbed.frames):
            filename = os.path.join(path, "{}.jpg".format(i))
            image = Image.fromarray(frame)
            image.save(filename)
        perturbed.cleanup()


class VideoFramer(Framer):
    def __init__(self, video_filename, r, p):
        self.video_filename = video_filename
        Framer.__init__(r, p)

    def fetch_frames(self):
        return read_video.VideoFrames(self.video_filename)


class GlobFramer(Framer):
    def __init__(self, glob, r, p):
        self.glob = glob
        Framer.__init__(r, p)

    def fetch_frames(self):
        return read_video.GlobFrames(self.glob)


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
