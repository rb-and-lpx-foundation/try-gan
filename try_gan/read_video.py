import cv2
import numpy as np
from glob import glob

from try_gan.image_files import bgr2rgb, open_jpg
from try_gan.perturb import Perturber


def perturbed_frames(frames, perturber):
    for frame in frames:
        yield perturber.perturb(frame), frame


def sample_from_generator(g, count, sample_count, r: np.random.RandomState):
    sample_count = min(sample_count, count)
    indices = r.choice(count, size=sample_count, replace=False)
    for i, x in enumerate(g):
        if i in indices:
            yield x


def concatenated_frames(frames, perturber):
    for perturbed, frame in perturbed_frames(frames, perturber):
        yield np.concatenate([frame, perturbed], axis=1)


class Frames:
    def __init__(self, frames, frame_count, fps):
        self.frames = frames
        self.frame_count = frame_count
        self.fps = fps

    def cleanup(self):
        raise NotImplementedError()


class WrappedFrames(Frames):
    def __init__(self, frames_impl: Frames, get_frames):
        self._frames_impl = frames_impl
        frames = get_frames(frames_impl.frames)
        Frames.__init__(self, frames, frames_impl.frame_count, frames_impl.fps)

    def cleanup(self):
        self._frames_impl.cleanup()


class PerturbedFrames(WrappedFrames):
    def __init__(self, frames_impl: Frames, perturber: Perturber):
        self.perturber = perturber

        def get_frames(frames):
            return map(perturber.perturb, frames)

        WrappedFrames.__init__(self, frames_impl, get_frames)


class SampledFrames(WrappedFrames):
    def __init__(self, frames_impl: Frames, sample_count, r: np.random.RandomState):
        sample_count = min(sample_count, frames_impl.frame_count)

        WrappedFrames.__init__(
            self,
            frames_impl,
            lambda frames: sample_from_generator(
                frames, frames_impl.frame_count, sample_count, r
            ),
        )


class VideoFrames(Frames):
    def __init__(self, filename):
        self.in_use = True
        self.vidcap = cv2.VideoCapture(filename)
        frame_count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.vidcap.get(cv2.CAP_PROP_FPS)

        def get_frames():
            success, image = self.vidcap.read()
            while success:
                success, frame = self.vidcap.read()
                if success:
                    yield bgr2rgb(frame)
                else:
                    self.cleanup()

        Frames.__init__(self, get_frames(), frame_count, fps)

    def cleanup(self):
        if self.in_use:
            self.vidcap.release()
            self.in_use = False


class GlobFrames(Frames):
    def __init__(self, g, fps, open_image_file=open_jpg):
        names = glob(g)
        names.sort()

        def frames_from_glob():
            for name in names:
                yield open_image_file(name)

        Frames.__init__(self, frames_from_glob(), len(names), fps)
