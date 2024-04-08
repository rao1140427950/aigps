import os
import numpy as np
import tensorflow as tf


PI = 3.14159

def prepare_inputs(images, periods, img_size=256, min_p=300., max_p=700.):
    if images.ndim == 2:
        images = images[np.newaxis, :, :, np.newaxis]
    elif images.ndim == 3:
        images = images[..., np.newaxis]
    images = tf.image.resize(images, (img_size, img_size))
    periods = np.reshape(periods, (-1, 1))
    images = images * 2. - 1.
    periods = (periods - min_p) / (max_p - min_p) * 2. - 1.
    # periods = np.tile(periods, (1, 2))
    return tf.cast(images, tf.float32), tf.cast(periods, tf.float32)


def process_outputs(tpower, rescale=5., do_ifft=False):
    tpower /= rescale
    real_part = tpower[..., 0]
    imag_part = tpower[..., 1]
    tpower = real_part + 1.j * imag_part
    if do_ifft:
        tpower = np.fft.ifft(tpower)
    return np.squeeze(np.abs(tpower)), np.squeeze(np.angle(tpower) % (2 * np.pi))


def get_base_dataset(data_path, batch_size=64, out_dim=201, img_size=256, shuffle=True, random_roll=True, cache=False):
    feature_description = {
        'pattern': tf.io.FixedLenFeature([], tf.string),
        'period': tf.io.FixedLenFeature((1,), tf.float32),
        't_power': tf.io.FixedLenFeature((out_dim,), tf.float32),
        't_phase': tf.io.FixedLenFeature((out_dim,), tf.float32),
    }

    def _parse_example_fcn(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        pattern = example['pattern']
        pattern = tf.image.decode_png(pattern, channels=1)
        pattern = tf.image.resize(pattern, (img_size, img_size))

        if random_roll:
            shift = tf.random.uniform(shape=(2,), minval=0, maxval=img_size, dtype=tf.int32)
            pattern = tf.roll(pattern, shift, axis=(0, 1))

        example['pattern'] = pattern
        return example

    data = tf.data.TFRecordDataset(data_path, num_parallel_reads=len(data_path))
    data = data.map(_parse_example_fcn)
    if cache:
        data = data.cache()
    if shuffle:
        data = data.shuffle(2000, reshuffle_each_iteration=True)
    data = data.batch(batch_size)
    return data


def generate_dataset_from_tfrecords(data_path, batch_size=64, out_dim=201, img_size=256, shuffle=True, random_roll=True,
                                    type_=None, out_seq_length=201, do_fft=False):

    def _batch_map_fcn(example, min_p=300, max_p=700, rescale_tpower=True):

        pattern = example['pattern']
        pattern = tf.cast(pattern, tf.float32)
        pattern = pattern * 2. - 1.

        t_power = example['t_power']  # (bsize, 201)
        t_phase = example['t_phase']
        t_power = tf.cast(t_power, tf.float32)
        t_phase = tf.cast(t_phase, tf.float32)
        real_part = t_power * tf.math.cos(t_phase)
        imag_part = t_power * tf.math.sin(t_phase)

        if do_fft:
            sig = tf.cast(real_part, tf.complex64) + tf.cast(1.j, tf.complex64) * tf.cast(imag_part, tf.complex64)
            fft = tf.signal.fft(sig)
            real_part = tf.math.real(fft)
            imag_part = tf.math.imag(fft)

        t_power = tf.concat([real_part[..., tf.newaxis], imag_part[..., tf.newaxis]], axis=-1)  # (bsize, 201, 2)
        if out_seq_length != 201:
            t_power = tf.image.resize(t_power[..., tf.newaxis], (out_seq_length, 2), method='bicubic')
            t_power = tf.squeeze(t_power)
        if rescale_tpower:
            t_power *= 5.

        period = example['period']
        period = tf.cast((period - min_p) / (max_p - min_p), tf.float32) * 2. - 1.  # (bsize, 1)
        if type_ == 'encoder_only':
            return (pattern, period), t_power
        else:
            tpower_input = t_power[:, :-1, :]  # (bsize, 201, 2)
            return (pattern, period, tpower_input), t_power

    data = get_base_dataset(data_path, batch_size, out_dim, img_size, shuffle, random_roll)
    data = data.map(_batch_map_fcn, num_parallel_calls=4)
    return data.prefetch(16)


def build_datasets(train_paths, val_paths, batch_size, image_size=256):
    train_dataset = generate_dataset_from_tfrecords(
        train_paths, batch_size=batch_size, img_size=image_size, shuffle=True, random_roll=True, out_seq_length=257,
        type_='encoder_only')
    val_dataset = generate_dataset_from_tfrecords(
        val_paths, batch_size=batch_size, img_size=image_size, shuffle=False, random_roll=False, out_seq_length=257,
        type_='encoder_only')
    return train_dataset, val_dataset


def config(gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


