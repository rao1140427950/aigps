"""
credit to:
    https://keras.io/examples/generative/ddim/
"""
import keras
import tensorflow as tf
from keras import layers
from tqdm import trange
from math import pi
import numpy as np


def diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    # diffusion times -> angles
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    # angles -> signal and noise rates
    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)
    # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

    return noise_rates, signal_rates


def sinusoidal_embedding(
        x,
        embedding_min_frequency=1.0,
        embedding_dims=256,
        embedding_max_frequency=1000.0,
):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=-1
    )
    return embeddings


class DiffusionWrapper(keras.Model):

    def __init__(
            self,
            diffussion_model,
            image_encoder=None,
            image_decoder=None,
            t_predictor=None,
            t_encoder=None,
            cond_type='c4shape_power',
            temb_ffdim=256,
            cond_ffdim=128,
            image_size=256,
            latent_size=32,
            latent_channels=2,
            latents_mean=(0.21755332, 0.5430553),
            latents_var=(0.0326933, 0.00500607),
            *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.network = diffussion_model
        self.ema_network = keras.models.clone_model(self.network)
        self.normalizer = keras.layers.Normalization(axis=-1, mean=latents_mean, variance=latents_var)
        self.denormalizer = keras.layers.Normalization(axis=-1, mean=latents_mean, variance=latents_var, invert=True)
        self.ema = 0.999
        self.temb_ffdim = temb_ffdim
        self.cond_ffdim = cond_ffdim
        self.image_size = image_size
        self.image_encoder = image_encoder
        self.image_decoder = image_decoder
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.t_predictor = t_predictor
        self.t_encoder = t_encoder
        self.cond_type = cond_type

    def call(self, inputs, training=None, mask=None):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        noisy_images, temb, conds = inputs
        pred_noises = network([noisy_images, temb] + conds, training=training)
        return pred_noises

    def denoise(self, inputs, noise_rates, signal_rates, training):
        noisy_images = inputs[0]
        # predict noise component and calculate the image component using it
        pred_noises = self.call(inputs, training=training)
        pred_images = (noisy_images - noise_rates[..., tf.newaxis, tf.newaxis] * pred_noises) / signal_rates[..., tf.newaxis, tf.newaxis]

        return pred_noises, pred_images

    @staticmethod
    def decouple_generated_sample(samples):
        samples = samples.copy()
        samples = np.squeeze(samples)
        p = np.mean(np.abs(samples), axis=(-1, -2))
        p = p * 400. + 300.
        samples[samples >= 0] = 1.
        samples[samples < 0] = 0.
        return samples, p

    def reverse_diffusion(self, initial_noise, diffusion_steps, conds):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        pred_images = None
        for step in trange(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1)) - step * step_size
            noise_rates, signal_rates = diffusion_schedule(diffusion_times)

            temb = sinusoidal_embedding(noise_rates ** 2, embedding_dims=self.temb_ffdim)

            pred_noises, pred_images = self.denoise(
                [noisy_images, temb, conds], noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times[..., tf.newaxis, tf.newaxis] - step_size
            next_noise_rates, next_signal_rates = diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps, conds, seed=None):
        # noise -> images -> denormalized images
        if seed is not None:
            if isinstance(seed, int):
                seed = (seed, seed)
            initial_noise = tf.random.stateless_normal(seed=seed,
                shape=(num_images, self.latent_size, self.latent_size, self.latent_channels))
        else:
            initial_noise = tf.random.normal(shape=(num_images, self.latent_size, self.latent_size, self.latent_channels))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps, conds)
        generated_images = self.denormalizer(generated_images)
        generated_images = self.image_decoder(generated_images)
        return generated_images.numpy()

    @staticmethod
    def random_drop(data, rate=0.3):
        def fcn1():
            return data
        def fcn2():
            return tf.zeros_like(data, dtype=tf.float32)
        return tf.cond(tf.greater(tf.random.uniform([], 0., 1.), rate), true_fn=fcn1, false_fn=fcn2)

    def prepare_conds(self, images, ps, labels=0, masks=None, dropout=False):
        if self.cond_type == 'c4shape':
            return [labels]

        if 'power' in self.cond_type:
            hpower, hphase = self.t_predictor(images, ps)
            if masks is None:
                masks = tf.ones_like(hpower, dtype=tf.float32)
            hpower_cond = self.t_encoder(hpower, masks)
            if dropout:
                hpower_cond = self.random_drop(hpower_cond)

            if self.cond_type == 'power':
                return [hpower_cond]
            elif self.cond_type == 'c4shape_power':
                return [labels, hpower_cond]
            elif self.cond_type == 'polarized_power':
                images_r90 = tf.image.rot90(images)
                vpower, vphase = self.t_predictor(images_r90, ps)
                vpower_cond = self.t_encoder(vpower, masks)
                if dropout:
                    vpower_cond = self.random_drop(vpower_cond)
                return [hpower_cond, vpower_cond]

        elif 'phase' in self.cond_type:
            hpower, hphase = self.t_predictor(images, ps)
            if masks is None:
                masks = tf.ones_like(hphase, dtype=tf.float32)
            hphase_cond = self.t_encoder(hphase, masks)
            if dropout:
                hphase_cond = self.random_drop(hphase_cond)


            if self.cond_type == 'phase':
                return [hphase_cond]
            elif self.cond_type == 'c4shape_phase':
                return [labels, hphase_cond]
            elif self.cond_type == 'polarized_phase':
                images_r90 = tf.image.rot90(images)
                vpower, vphase = self.t_predictor(images_r90, ps)
                vphase_cond = self.t_encoder(vphase, masks)
                if dropout:
                    vphase_cond = self.random_drop(vphase_cond)
                return [hphase_cond, vphase_cond]
        else:
            raise ValueError('Unknown input for `cond_type`.')


    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        (images, labels), masks = images
        batch_size = tf.shape(images)[0]

        p = tf.random.uniform(minval=300., maxval=700., shape=(batch_size, 1), dtype=tf.float32)

        conds = self.prepare_conds(images, p, labels, masks, dropout=True)

        p_range = (p - 300.) / 400.
        images = images * 2. - 1.
        images = images * tf.reshape(p_range, (-1, 1, 1, 1))
        images = self.image_encoder(images)

        images = self.normalizer(images)
        noises = tf.random.normal(shape=tf.shape(images))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
        noise_rates, signal_rates = diffusion_schedule(diffusion_times)
        temb = sinusoidal_embedding(noise_rates ** 2, embedding_dims=self.temb_ffdim)
        # mix the images with noises accordingly
        noisy_images = signal_rates[..., tf.newaxis, tf.newaxis] * images + noise_rates[..., tf.newaxis, tf.newaxis] * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                [noisy_images, temb, conds], noise_rates, signal_rates, training=True
            )

            loss = self.loss(noises, pred_noises)  # used for training
            # loss = tf.clip_by_value(loss, -500., 500.)

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return self.compute_metrics(noisy_images, noises, pred_noises, None)

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        self.network.save(filepath, overwrite, save_format, **kwargs)
        self.ema_network.save(filepath + '_ema.h5', overwrite, save_format, **kwargs)

    def save_weights(
        self, filepath, overwrite=True, save_format=None, options=None
    ):
        self.network.save_weights(filepath, overwrite, save_format, options)
        self.ema_network.save_weights(filepath + '_ema.h5', overwrite, save_format, options)

    def load_weights(
        self, filepath, skip_mismatch=False, by_name=False, options=None
    ):
        self.network.load_weights(filepath, skip_mismatch, by_name, options)
        self.ema_network.load_weights(filepath + '_ema.h5', skip_mismatch, by_name, options)
