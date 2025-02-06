from models.common.dataset import config
if __name__ == '__main__':
    config('0, 1, 2, 3, 4, 5, 6, 7')

import tensorflow as tf
from models.common.dataset import PatternsDataset
from diffusion.ldm import latent_diffusion
from models.image_cvae import batch_encode_shapes
from models.power_encoder import batch_encode_powers
from models.property_predictor import batch_predict_property
from diffusion.ddim_sampler import DiffusionWrapper
from models.common.schedule import CustomSchedule


def get_dataset(
        batch_size=48,
        image_size=256,
):
    dataset = PatternsDataset(tfrecord_path=[
        './srcs/random_c4_shapes_124k.tfrecords',
        './srcs/random_shapes_80k.tfrecords',
    ], image_size=image_size, random_roll=True, random_invert=True, random_mask=True)
    dataset = dataset.generate_dataset_from_tfrecords(batch_size=batch_size, return_label=True)

    return dataset


def train(
        batch_size=16 * 8,
        num_epochs=40,
        model_name='sd-ddim-c4shape-power-vconcat',
        cond_type='c4shape_power',
):

    dataset = get_dataset(batch_size=batch_size)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        diffusion_core = latent_diffusion(cond_type=cond_type)

        diffusion_wrapper = DiffusionWrapper(
            diffussion_model=diffusion_core,
            image_encoder=batch_encode_shapes,
            t_predictor=batch_predict_property,
            t_encoder=batch_encode_powers,
            cond_type=cond_type,
        )
        diffusion_wrapper.compile(
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
            optimizer=tf.keras.optimizers.Adam(learning_rate=CustomSchedule(), weight_decay=1e-4),
            metrics=['mse'],
        )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./checkpoints/' + model_name,
        write_graph=False,
        write_images=False,
        update_freq=10,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/ckpt_{}.h5'.format(model_name),
        monitor='loss',
        save_weights_only=True,
        save_freq='epoch',
    )

    diffusion_wrapper.fit(
        dataset,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[tensorboard, checkpoint]
    )

    diffusion_wrapper.save_weights('./checkpoints/{}.h5'.format(model_name))


if __name__ == '__main__':
    train()
