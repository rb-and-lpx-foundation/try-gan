# Modified from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb

import os

import tensorflow as tf

from try_gan.pipeline import Pipeline


class FilePipeline(Pipeline):
    def __init__(self, path):
        self.PATH = path

    def _train_filenames(self):
        raise NotImplementedError()

    @property
    def train_filenames(self):
        return self._train_filenames()

    def _test_filenames(self):
        raise NotImplementedError()

    @property
    def test_filenames(self):
        return self._test_filenames()

    def load_raw(self, image_file):
        raise NotImplementedError()

    def load(self, image_file):
        image = self.load_raw(image_file)

        w = tf.shape(image)[1]

        w = w // 2
        real_image = image[:, :w, :]
        input_image = image[:, w:, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def load_image_train(self, image_file):
        input_image, real_image = self.load(image_file)
        return self.process_image_train(input_image, real_image)

    def load_image_test(self, image_file):
        input_image, real_image = self.load(image_file)
        return self.process_image_test(input_image, real_image)

    def make_train(self):
        train_dataset = tf.data.Dataset.list_files(self.train_filenames)
        train_dataset = train_dataset.map(
            self.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE)
        return train_dataset.batch(self.BATCH_SIZE)

    def make_test(self):
        test_dataset = tf.data.Dataset.list_files(self.test_filenames)
        test_dataset = test_dataset.map(self.load_image_test)
        return test_dataset.batch(self.BATCH_SIZE)


class JpgFilePipeline(FilePipeline):
    def __init__(self, path):
        FilePipeline.__init__(self, path)

    def _train_filenames(self):
        path = os.path.join(self.PATH, "train")
        return os.path.join(path, "*.jpg")

    def _test_filenames(self):
        path = os.path.join(self.PATH, "test")
        return os.path.join(path, "*.jpg")

    def load_raw(self, image_file):
        image = tf.io.read_file(image_file)
        return tf.image.decode_jpeg(image)
