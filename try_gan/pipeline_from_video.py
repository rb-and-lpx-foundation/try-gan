import numpy as np
from PIL import Image
import os

from try_gan import read_video
from try_gan import perturb
from try_gan.pipeline_from_files import JpgFilePipeline as Pipeline


class Framer:
    r = np.random.RandomState(42)
    sigma = 0.1
    g = perturb.GaussPerturber(r, sigma=sigma)
    s_vs_p = 0.3
    amount = 0.1
    snp = perturb.SNPPerturber(r, s_vs_p=s_vs_p, amount=amount)
    ops = [perturb.Normalizer(), g, snp, perturb.Discretizer()]
    p = perturb.CompositePerturber(ops)
    video = "data/hiragana128.mov"

    def write_samples(self, path, n):
        frame_count, frames = read_video.open_video(self.video)
        sampler = read_video.sample_from_generator(frames, frame_count, n, self.r)
        for i, frame in enumerate(read_video.concatenated_frames(sampler, self.p)):
            filename = os.path.join(path, "{}.jpg".format(i))
            image = Image.fromarray(frame)
            image.save(filename)



class VideoFramePipeline(Pipeline):
    def __init__(self, path):
        Pipeline.__init__(self, path)
        self.framer = Framer()


    def make_train(self):
        path = os.path.join(self.PATH, "train")
        self.framer.write_samples(path, 400)
        return Pipeline.make_train(self)

    def make_test(self):
        path = os.path.join(self.PATH, "test")
        self.framer.write_samples(path, 40)
        return Pipeline.make_test(self)
