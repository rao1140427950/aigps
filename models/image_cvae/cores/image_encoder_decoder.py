"""
credit to:
    https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion
"""
import keras
from keras import layers
from models.image_cvae.cores.layers import PaddedConv2D, ResnetBlock, AttentionBlock
from models.common.dataset import PatternsDataset
import tensorflow as tf


class ImageEncoder(keras.Sequential):
    """ImageEncoder is the VAE Encoder for StableDiffusion."""

    def __init__(self, image_size=256, image_channels=1, output_channels=2, base_ffdim=32):
        super().__init__(
            [
                layers.Input((image_size, image_size, image_channels)),
                PaddedConv2D(base_ffdim, 3, padding=1),
                ResnetBlock(base_ffdim),
                ResnetBlock(base_ffdim),
                PaddedConv2D(base_ffdim, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(base_ffdim * 2),
                ResnetBlock(base_ffdim * 2),
                PaddedConv2D(base_ffdim * 2, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(base_ffdim * 4),
                ResnetBlock(base_ffdim * 4),
                PaddedConv2D(base_ffdim * 4, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(base_ffdim * 4),
                ResnetBlock(base_ffdim * 4),

                ResnetBlock(base_ffdim * 4),
                AttentionBlock(base_ffdim * 4),
                ResnetBlock(base_ffdim * 4),
                layers.GroupNormalization(epsilon=1e-5),
                layers.Activation("swish"),
                PaddedConv2D(output_channels * 2, 3, padding=1),
                PaddedConv2D(output_channels, 1),
                layers.Activation("sigmoid"),
            ]
        )


class ImageDecoder(keras.Sequential):
    def __init__(self, input_size=32, input_channels=2, output_channels=1, base_ffdim=32):
        super().__init__(
            [
                layers.Input((input_size, input_size, input_channels)),
                PaddedConv2D(input_channels, 1),
                PaddedConv2D(base_ffdim * 4, 3, padding=1),
                ResnetBlock(base_ffdim * 4),
                AttentionBlock(base_ffdim * 4),
                ResnetBlock(base_ffdim * 4),
                ResnetBlock(base_ffdim * 4),
                ResnetBlock(base_ffdim * 4),
                ResnetBlock(base_ffdim * 4),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(base_ffdim * 4, 3, padding=1),
                ResnetBlock(base_ffdim * 4),
                ResnetBlock(base_ffdim * 4),
                ResnetBlock(base_ffdim * 4),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(base_ffdim * 4, 3, padding=1),
                ResnetBlock(base_ffdim * 2),
                ResnetBlock(base_ffdim * 2),
                ResnetBlock(base_ffdim * 2),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(base_ffdim * 2, 3, padding=1),
                ResnetBlock(base_ffdim),
                ResnetBlock(base_ffdim),
                ResnetBlock(base_ffdim),
                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(output_channels, 3, padding=1),
            ]
        )


class EncoderDecoder(keras.Model):

    def __init__(self, encoder=None, decoder=None, image_size=256, image_channels=1, latent_size=32,
                 latent_channels=2, base_ffdim=32, random_scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        if self.encoder is None:
            self.encoder = ImageEncoder(image_size, image_channels, latent_channels, base_ffdim)
        self.decoder = decoder
        if self.decoder is None:
            self.decoder = ImageDecoder(latent_size, latent_channels, image_channels, base_ffdim)
        self.random_scale = random_scale
        self.noise_strength = self.add_weight(name='noise_strength', shape=None, dtype=tf.float32,
                                              initializer=tf.keras.initializers.Zeros(), trainable=True)

    def call(self, inputs, training=None, mask=None):
        latents = self.encoder(inputs, training)
        noise = tf.random.normal(shape=tf.shape(latents), dtype=tf.float32)
        latents = latents + noise * self.noise_strength
        recover = self.decoder(latents, training)
        return recover

    def apply_random_scale(self, data):
        batch_size = tf.shape(data)[0]
        if self.random_scale:
            p = tf.random.uniform(minval=300., maxval=700., shape=(batch_size,), dtype=tf.float32)
            p_range = (p - 300.) / 400.
            images = data * tf.reshape(p_range, (-1, 1, 1, 1))
        else:
            images = data
        return images

    def train_step(self, data):
        images = self.apply_random_scale(data)

        with tf.GradientTape() as tape:
            recover = self.call(images, training=True)
            loss = self.loss(images, recover)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return {'loss': loss}

    def load_weights_separately(
        self, filepath, skip_mismatch=False, by_name=False, options=None
    ):
        self.encoder.load_weights(filepath + '_encoder.h5', skip_mismatch, by_name, options)
        self.decoder.load_weights(filepath + '_decoder.h5', skip_mismatch, by_name, options)

    def save_weights_separately(
        self, filepath, overwrite=True, save_format=None, options=None
    ):
        self.encoder.save_weights(filepath + '_encoder.h5', overwrite, save_format, options)
        self.decoder.save_weights(filepath + '_decoder.h5', overwrite, save_format, options)

    def save_separately(self, filepath, overwrite=True, save_format=None, **kwargs):
        self.encoder.save(filepath + '_encoder.h5', overwrite, save_format, **kwargs)
        self.decoder.save(filepath + '_decoder.h5', overwrite, save_format, **kwargs)


class ExportWrapper(tf.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)])
    def __call__(self, shapes):
        latents = self.encoder(shapes)
        return self.decoder(latents)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)])
    def encode(self, shapes):
        return self.encoder(shapes)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 32, 32, 2), dtype=tf.float32)])
    def decode(self, shapes):
        return self.decoder(shapes)


def export_model(weights, save_path):
    vae = EncoderDecoder(image_size=256)
    vae.load_weights_separately(weights)

    vae_wrapper = ExportWrapper(vae.encoder, vae.decoder)

    shape = tf.random.normal((1, 256, 256, 1), dtype=tf.float32)

    vae_wrapper(shape)
    lts = vae_wrapper.encode(shape)
    _ = vae_wrapper.decode(lts)

    tf.saved_model.save(vae_wrapper, save_path)


def compute_mean_var():
    image_size = 256

    model = EncoderDecoder(image_size=image_size)
    model.load_weights_separately('./checkpoints/image_encoder_decoder/image_vae.h5')

    dataset = PatternsDataset(tfrecord_path=[
        './dataset/random_c4_shapes_124k.tfrecords',
        './dataset/random_shapes_50k.tfrecords',
    ], image_size=image_size, random_roll=True, random_invert=True)
    dataset = dataset.generate_dataset_from_tfrecords(batch_size=16, return_label=False)

    def map_fcn(images):
        images = model.apply_random_scale(images)
        with tf.device('/device:gpu:0'):
            latents = model.encoder(images, training=False)
        return latents

    data = dataset.map(map_fcn)

    nomalizer = tf.keras.layers.Normalization(axis=-1)
    nomalizer.adapt(data)

    print(nomalizer.mean, nomalizer.variance)


def train():
    image_size = 256
    batch_size = 8 * 8
    epochs = 16
    model_name = 'image_vae.h5'
    work_dir = './checkpoints/image_encoder_decoder/'
    log_dir = work_dir + model_name
    output_model_file = work_dir + model_name
    checkpoint_path = work_dir + 'checkpoint-' + model_name


    dataset = PatternsDataset(tfrecord_path=[
        './dataset/random_c4_shapes_124k.tfrecords',
        './dataset/random_shapes_50k.tfrecords',
    ], image_size=image_size, random_roll=True, random_invert=True)
    dataset = dataset.generate_dataset_from_tfrecords(batch_size=batch_size, return_label=False)

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_graph=False,
        write_images=False,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='loss',
        save_weights_only=True,
        save_freq='epoch',
    )
    def scheduler(epoch, lr):
        if epoch in [8, 12]:
            return lr * 0.1
        else:
            return lr
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = EncoderDecoder(image_size=image_size)
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.SUM),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, weight_decay=1e-4),
        )

    model.load_weights_separately(output_model_file)

    model.fit(
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tensorboard, checkpoint, lr_scheduler]
    )
    model.save_separately(output_model_file)

