from glob import glob

from try_gan import image_files
from try_gan.framer import Framer


def frames_from_glob(g):
    names = glob(g)

    def g():
        for name in names:
            yield image_files.open_jpg(name)

    return len(names), g


class GlobFramer(Framer):
    def __init__(self, glob, r=None, p=None):
        Framer.__init__(self, r=r, p=p)
        self.glob = glob

    def get_frames(self):
        frame_count, g = frames_from_glob(self.glob)
        return frame_count, g()
