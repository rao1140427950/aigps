import tensorflow as tf
import os.path

__current_dir = os.path.dirname(__file__)
__models_dir = os.path.join(__current_dir, '../exported_models')


DEFAULT_CVAE = tf.saved_model.load(os.path.join(__models_dir, 'image_cvae'))
LATENTS_MEAN = (0.21755332, 0.5430553)
LATENTS_VAR = (0.0326933, 0.00500607)


def batch_encode_shapes(shapes, model=DEFAULT_CVAE):
    shapes = tf.image.resize(shapes, (256, 256))
    return model.encode(shapes)


def batch_decode_shapes(latents, model=DEFAULT_CVAE):
    return model.decode(latents)


if __name__ == '__main__':
    import scipy.io as sio
    import numpy as np
    import matplotlib.pyplot as plt

    mat = sio.loadmat('../../srcs/data_FDTD.mat')
    shape = (mat['pattern'] * 2. - 1.) * 0.5

    shape_tf = tf.cast(shape[np.newaxis, ..., np.newaxis], tf.float32)

    lts = batch_encode_shapes(shape_tf)
    shape_recover = batch_decode_shapes(lts)
    shape_recover = np.squeeze(shape_recover.numpy())

    plt.subplot(1, 2, 1)
    plt.imshow(shape, vmin=-1., vmax=1., cmap='gray')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(shape_recover, vmin=-1., vmax=1., cmap='gray')
    plt.title('Recovered')
    plt.show()