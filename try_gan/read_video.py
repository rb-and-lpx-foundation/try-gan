import cv2


def frames(video):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    while success:
        success, frame = vidcap.read()
        if success:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def perturbed_frames(frames, perturber):
    for frame in frames:
        yield perturber.perturb(frame), frame
