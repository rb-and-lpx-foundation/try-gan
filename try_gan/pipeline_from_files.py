# Modified from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb

import tensorflow as tf


class Pipeline:
    BUFFER_SIZE = 400
    BATCH_SIZE = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    def resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(
            input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        real_image = tf.image.resize(
            real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        return input_image, real_image

    def random_crop(self, input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 3]
        )

        return cropped_image[0], cropped_image[1]

    def normalize(self, input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    @tf.function()
    def random_jitter(self, input_image, real_image):
        # resizing to 286 x 286 x 3
        input_image, real_image = self.resize(input_image, real_image, 286, 286)

        # randomly cropping to 256 x 256 x 3
        input_image, real_image = self.random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def process_image_train(self, input_image, real_image):
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def process_image_test(self, input_image, real_image):
        input_image, real_image = self.resize(
            input_image, real_image, self.IMG_HEIGHT, self.IMG_WIDTH
        )
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def make_train(self):
        raise NotImplementedError()

    def make_test(self):
        raise NotImplementedError()

    def make_datasets(self):
        return self.make_train(), self.make_test()


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
        return self.PATH + "train/*.jpg"

    def _test_filenames(self):
        return self.PATH + "test/*.jpg"

    def load_raw(self, image_file):
        image = tf.io.read_file(image_file)
        return tf.image.decode_jpeg(image)
