import numpy as np
from models.common.dataset import shapes_tphase_dataset
from models.power_encoder.cores.tpower_mae import Encoder, Decoder, MAETrainer
import tensorflow as tf
import os


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


def train():
    batch_size = 64 * 8
    epochs = 160
    model_name = 'tphase_mae_m.h5'
    work_dir = './checkpoints/tphase_mae/'
    output_model_file = work_dir + model_name
    checkpoint_path = work_dir + 'checkpoint-' + model_name

    dataset = shapes_tphase_dataset(batch_size)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='loss',
        save_weights_only=True,
        save_freq='epoch',
    )
    def scheduler(epoch, lr):
        if epoch in [80, 120]:
            return lr * 0.1
        else:
            return lr
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        encoder = Encoder(
            num_heads=4,
            num_layers=4,
            ffdim=128,
            mlp_units=(256, 128),
            dropout=0.1,
            input_ffdim=2,
            input_length=201,
        )
        decoder = Decoder(
            num_heads=4,
            num_layers=2,
            ffdim=64,
            mlp_units=(128, 64),
            dropout=0.1,
            output_ffdim=2,
        )

        model = MAETrainer(encoder, decoder)
        d = tf.random.normal((4, 201, 2))
        m = tf.random.uniform((4, 201), minval=0., maxval=1.)
        model([d, m])
        model.load_weights_separately('./checkpoints/tphase_mae/tphase_mae_m.h5')
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.SUM),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, weight_decay=1e-4),
        )

    model.fit(
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, lr_scheduler]
    )
    model.save_weights_separately(output_model_file)


class ExportWrapper(tf.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def batch_data_map_fcn(data):
        cosdata = tf.math.cos(data)
        sindata = tf.math.sin(data)
        data = tf.stack([cosdata, sindata], axis=-1)
        return data

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 201), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 201))])
    def __call__(self, powers, masks):
        powers = self.batch_data_map_fcn(powers)
        latents = self.encoder([powers, masks])
        return self.decoder(latents)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 201), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 201))])
    def encode(self, powers, masks):
        powers = self.batch_data_map_fcn(powers)
        return self.encoder([powers, masks])

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 201, 128), dtype=tf.float32)])
    def decode(self, latents):
        data = self.decoder(latents) * np.pi
        cos = data[..., 0]
        sin = data[..., 1]
        cplx = tf.complex(cos, sin)
        return tf.math.angle(cplx)


def export_model():
    encoder = Encoder(
        num_heads=4,
        num_layers=4,
        ffdim=128,
        mlp_units=(256, 128),
        dropout=0.1,
        input_ffdim=2,
        input_length=201,
    )
    decoder = Decoder(
        num_heads=4,
        num_layers=2,
        ffdim=64,
        mlp_units=(128, 64),
        dropout=0.1,
        output_ffdim=2,
    )

    model = MAETrainer(encoder, decoder)
    d = tf.random.normal((4, 201, 2))
    m = tf.random.uniform((4, 201), minval=0., maxval=1.)
    model([d, m])
    model.load_weights_separately('./checkpoints/tphase_mae/tphase_mae_m.h5')

    d = tf.random.normal((4, 201))
    mae_wrapper = ExportWrapper(model.encoder, model.decoder)
    mae_wrapper(d, m)
    lts = mae_wrapper.encode(d, m)
    mae_wrapper.decode(lts)

    tf.saved_model.save(mae_wrapper, './checkpoints/phase_mae')


