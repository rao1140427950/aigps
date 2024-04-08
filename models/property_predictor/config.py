import tensorflow as tf
import os


__current_dir = os.path.dirname(__file__)
__models_dir = os.path.join(__current_dir, '../exported_models')


T_PREDICTOR_4LAYER = tf.saved_model.load(os.path.join(__models_dir, 'encoder_only_4layer'))


def batch_prepare_inputs_tf(images, periods, img_size=256, min_p=300., max_p=700.):
    images = tf.image.resize(images, (img_size, img_size))
    periods = tf.reshape(periods, (-1, 1))
    images = images * 2. - 1.
    periods = (periods - min_p) / (max_p - min_p) * 2. - 1.
    return tf.cast(images, tf.float32), tf.cast(periods, tf.float32)


def batch_process_outputs_tf(tpower, rescale=5., n_samples=201):
    tpower /= rescale
    if n_samples != 257:
        tpower = tf.squeeze(tf.image.resize(tpower[..., tf.newaxis], (n_samples, 2)))
    real_part = tpower[..., 0]
    imag_part = tpower[..., 1]
    tpower = tf.complex(real_part, imag_part)
    return tf.squeeze(tf.math.abs(tpower)), tf.squeeze(tf.math.angle(tpower))

def batch_predict_property(shapes, periods, model=T_PREDICTOR_4LAYER):
    images, periods = batch_prepare_inputs_tf(shapes, periods)
    pred = model(images, periods)
    tpower_pred, tphase_pred = batch_process_outputs_tf(pred)
    return tpower_pred, tphase_pred


if __name__ == '__main__':
    import scipy.io as sio
    import numpy as np
    import matplotlib.pyplot as plt

    mat = sio.loadmat('../../srcs/data_FDTD.mat')
    shape = mat['pattern']
    period = mat['period']
    power = np.squeeze(mat['T_power'])
    phase = np.squeeze(mat['T_phase'])
    # phase %= (2 * np.pi)

    period = tf.cast(period, tf.float32)
    shape = tf.cast(shape[np.newaxis, ..., np.newaxis], tf.float32)

    batch_size = 1

    shape = tf.tile(shape, (batch_size, 1, 1, 1))
    period = tf.tile(period, (batch_size, 1))

    power_pred, phase_pred = batch_predict_property(shape, period)
    power_pred = np.squeeze(power_pred.numpy())
    phase_pred = np.squeeze(phase_pred.numpy())

    sprange0 = np.linspace(400, 800, 201)

    plt.subplot(2, 1, 1)
    plt.plot(sprange0, power_pred)
    plt.plot(sprange0, power, '--')
    plt.title('T-Power')
    plt.subplot(2, 1, 2)
    plt.plot(sprange0, phase_pred)
    plt.plot(sprange0, phase, '--')
    plt.title('T-Phase')
    plt.show()