import cv2
import numpy as np

from try_gan.image_files import bgr2rgb


def frames(vidcap):
    success, image = vidcap.read()
    while success:
        success, frame = vidcap.read()
        if success:
            yield bgr2rgb(frame)
        else:
            vidcap.release()


def perturbed_frames(frames, perturber):
    for frame in frames:
        yield perturber.perturb(frame), frame


def open_video(video):
    vidcap = cv2.VideoCapture(video)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    return frame_count, frame_rate, frames(vidcap)


def sample_from_generator(g, count, sample_count, r: np.random.RandomState):
    indices = r.choice(count, size=sample_count, replace=False)
    for i, x in enumerate(g):
        if i in indices:
            yield x


def concatenated_frames(frames, perturber):
    for perturbed, frame in perturbed_frames(frames, perturber):
        yield np.concatenate([frame, perturbed], axis=1)
