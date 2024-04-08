import tensorflow as tf
import os.path
import numpy as np

__current_dir = os.path.dirname(__file__)
__models_dir = os.path.join(__current_dir, '../exported_models')


DEFAULT_MAE = tf.saved_model.load(os.path.join(__models_dir, 'power_mae'))


def batch_encode_powers(powers, masks, model=DEFAULT_MAE):
    return model.encode(powers, masks)


def batch_decode_powers(latents, model=DEFAULT_MAE):
    return model.decode(latents)


if __name__ == '__main__':
    import scipy.io as sio
    import matplotlib.pyplot as plt

    mat = sio.loadmat('../../srcs/data_FDTD.mat')
    shape = mat['pattern']
    period = mat['period']
    power = np.squeeze(mat['T_power'])
    mask = np.zeros_like(power)
    mask[50:150] = 1.
    power_tf = tf.cast(power[np.newaxis, ...], tf.float32)
    mask_tf = tf.cast(mask[np.newaxis, ...], tf.float32)


    lts = batch_encode_powers(power_tf, mask_tf)
    power_rcv = batch_decode_powers(lts).numpy().squeeze()

    lts = lts.numpy().squeeze()

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.plot(power)
    plt.plot(power_rcv[:, 0])
    plt.plot(mask, '--')
    plt.ylim([0, 1.2])
    plt.legend(['Orignal', 'Recovered', 'Mask'])

    plt.subplot(1, 2, 2)
    plt.imshow(lts)
    plt.show()
