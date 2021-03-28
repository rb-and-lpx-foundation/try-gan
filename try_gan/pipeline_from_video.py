from try_gan import read_video
from try_gan.framer import Framer


class VideoFramer(Framer):
    def __init__(self, video, r=None, p=None):
        Framer.__init__(self, r=r, p=p)
        self.video = video

    def get_frames(self):
        frame_count, _, frames = read_video.open_video(self.video)
        return frame_count, frames
