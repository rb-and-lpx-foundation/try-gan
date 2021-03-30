import cv2

from try_gan import image_files


def write_video(frames, path, rate=30.0, size=(256, 256)):
    frame_width, frame_height = size
    # Define the codec and create VideoWriter object for mp4
    out = cv2.VideoWriter(
        "{}.mp4".format(path),
        cv2.VideoWriter_fourcc(*"MP4V"),
        rate,
        (frame_width, frame_height),
    )

    for i, frame in enumerate(frames):
        out.write(image_files.rgb2bgr(frame))

    out.release()


def batch_videos(
    frames, total_frame_count, path, chunk_frame_count=3600, rate=30.0, size=(256, 256)
):
    def frame_chunk():
        for i, frame in enumerate(frames):
            yield frame
            if i >= chunk_frame_count - 1:
                break

    i = 0
    while total_frame_count > chunk_frame_count:
        if total_frame_count < 2 * chunk_frame_count:
            chunk_frame_count = total_frame_count
            total_frame_count = 0
        else:
            total_frame_count -= chunk_frame_count
        i += 1
        write_video(frame_chunk(), "{}_{}".format(path, i), rate=rate, size=size)
