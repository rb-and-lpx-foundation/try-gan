import tensorflow as tf

import os
import time
import datetime


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class Logger:
    def __init__(self, log_dir="logs", fit_dir="fit", filename=None):
        if filename is None:
            filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(log_dir, fit_dir)
        filename = os.path.join(path, filename)
        self.summary_writer = tf.summary.create_file_writer(filename)


class Checkpoint:
    def __init__(
        self, checkpoint_dir=None, prefix=None
    ):
        if checkpoint_dir is None:
            checkpoint_dir = "training_checkpoints"
        if prefix is None:
            prefix = "chkpt"
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.prefix = prefix

    @property
    def checkpoint_prefix(self):
        return os.path.abspath(os.path.join(self.checkpoint_dir, self.prefix))

    def save(self):
        raise NotImplementedError("To be implemented")

    def restore(self):
        raise NotImplementedError("To be implemented")


class Pix2pixCheckpoint(Checkpoint):
    def __init__(
        self,
        generator_optimizer,
        discriminator_optimizer,
        generator,
        discriminator,
        checkpoint_dir="training_checkpoints",
        prefix="chkpt",
    ):
        Checkpoint.__init__(
            self, checkpoint_dir=checkpoint_dir, prefix=prefix
        )
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator,
        )

    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))


class Pix2pix:
    OUTPUT_CHANNELS = 3
    LAMBDA = 100

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __init__(self, check=None, logger=None):
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        if logger is None:
            logger = Logger()
        self.logger = logger
        self.check = self.make_check(check)

    def make_check(self, checkpoint_dir=None):
        return Pix2pixCheckpoint(
            self.generator_optimizer,
            self.discriminator_optimizer,
            self.generator,
            self.discriminator,
            checkpoint_dir=checkpoint_dir,
        )

    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            downsample(128, 4),  # (bs, 64, 64, 128)
            downsample(256, 4),  # (bs, 32, 32, 256)
            downsample(512, 4),  # (bs, 16, 16, 512)
            downsample(512, 4),  # (bs, 8, 8, 512)
            downsample(512, 4),  # (bs, 4, 4, 512)
            downsample(512, 4),  # (bs, 2, 2, 512)
            downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            upsample(512, 4),  # (bs, 16, 16, 1024)
            upsample(256, 4),  # (bs, 32, 32, 512)
            upsample(128, 4),  # (bs, 64, 64, 256)
            upsample(64, 4),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0.0, 0.02)
        last = tf.keras.layers.Conv2DTranspose(
            self.OUTPUT_CHANNELS,
            4,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            activation="tanh",
        )  # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(
            tf.ones_like(disc_generated_output), disc_generated_output
        )

        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0.0, 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name="input_image")
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name="target_image")

        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

        down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(
            512, 4, strides=1, kernel_initializer=initializer, use_bias=False
        )(
            zero_pad1
        )  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
            zero_pad2
        )  # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(
            tf.zeros_like(disc_generated_output), disc_generated_output
        )

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    @tf.function
    def train_step(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator(
                [input_image, gen_output], training=True
            )

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                disc_generated_output, gen_output, target
            )
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        with self.logger.summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=epoch)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=epoch)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=epoch)
            tf.summary.scalar("disc_loss", disc_loss, step=epoch)

    def fit(self, epochs, pipeline, starting_epoch=0, cb=(lambda test_ds, epoch: None)):
        for i in range(epochs):
            epoch = starting_epoch + i
            train_ds = pipeline.make_train()
            test_ds = pipeline.make_test()

            start = time.time()

            cb(test_ds, epoch)

            # Train
            for n, (input_image, target) in train_ds.enumerate():
                print(".", end="")
                if (n + 1) % 100 == 0:
                    print()
                self.train_step(input_image, target, epoch)

            print()

        self.check.save()
        print(
            "Time taken for epoch {} is {} sec\n".format(epoch + 1, time.time() - start)
        )
