from matplotlib import pyplot as plt
from IPython import display
import numpy as np

from try_gan import read_video


def show_frames(frames: read_video.Frames, first_frame, n=10):
    print("{} total frames at {} fps".format(frames.frame_count, frames.fps))
    frame = None
    for i, frame in enumerate(frames.frames):
        if i >= first_frame:
            plt.figure()
            plt.imshow(frame)
        if i >= first_frame + n:
            break
    frames.cleanup()
    if frames.frame_count:
        return frame


def show_sampled_frames(frames: read_video.Frames, r: np.random.RandomState, n=10):
    samples = read_video.sample_from_generator(frames, n, r)
    frame = None
    for frame in samples.frames:
        plt.figure()
        plt.imshow(frame)
    frames.cleanup()
    return frame


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.show()


def show_cdgan_images(model, test_ds, epoch):
    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(2):
        generate_images(model.generator, example_input, example_target)
    print("Epoch: ", epoch)
