import tensorflow as tf
import os.path

__current_dir = os.path.dirname(__file__)
__models_dir = os.path.join(__current_dir, '../exported_models')


DEFAULT_MAE = tf.saved_model.load(os.path.join(__models_dir, 'phase_mae'))


def batch_encode_phases(phases, masks, model=DEFAULT_MAE):
    # phases = phases % (np.pi * 2) - np.pi
    return model.encode(phases, masks)


def batch_decode_phases(latents, model=DEFAULT_MAE):
    data = model.decode(latents)
    return data


