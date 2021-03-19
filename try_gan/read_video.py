import cv2

from try_gan.image_files import bgr2rgb


def frames(video):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    while success:
        success, frame = vidcap.read()
        if success:
            yield bgr2rgb(frame)


def perturbed_frames(frames, perturber):
    for frame in frames:
        yield perturber.perturb(frame), frame
