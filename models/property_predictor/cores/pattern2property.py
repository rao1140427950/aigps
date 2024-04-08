import tensorflow as tf
from models.common.transformer import EncoderPreprocessing, Encoder
from keras import layers
from keras.metrics import MeanSquaredError, CosineSimilarity
from models.property_predictor.cores.utils import build_datasets
from models.common.schedule import CustomSchedule


class EncoderOnly(tf.keras.Model):

    def __init__(
            self,
            num_layers=8,
            d_model=256,
            num_heads=8,
            image_size=256,
            patch_size=16,
            dropout=0.1,
            out_ffdim=2,
            trainable_position_emb=False,
            rescale=False,
            period_token=True,
    ):
        super(EncoderOnly, self).__init__()
        self.encoder_preprocesssing = EncoderPreprocessing(
            image_size=image_size,
            patch_size=patch_size,
            emb_size=d_model,
            period_token=period_token,
            trainable_positional_embedding=trainable_position_emb,
            rescale=rescale,
        )
        self.encoder = Encoder(num_layers, num_heads, d_model, dropout)
        self.final_layer = layers.Dense(out_ffdim)
        self.seq_length = self.encoder_preprocesssing.npos

    def call(self, inputs, training=None, mask=None):
        x = self.encoder_preprocesssing(inputs)
        x = self.encoder(x)
        x = self.final_layer(x)
        return x


def train():
    image_size = 256
    ffdim = 256
    num_layers = 4

    batch_size = 64 * 8
    epochs = 160 * 4
    model_name = 'encoderonly-4layer'
    work_dir = './checkpoints/'
    log_dir = work_dir + model_name
    output_model_file = work_dir + model_name + '.h5'
    checkpoint_path = work_dir + 'checkpoint-' + model_name + '.h5'

    trainset, valset = build_datasets(
        train_paths=['./srcs/datasets/dataset_220nm_random_c4.tfrecords',
                     './srcs/datasets/dataset_220nm_random.tfrecords',],
        val_paths=['./srcs/datasets/dataset_220nm_val.tfrecords'], batch_size=batch_size, image_size=image_size)

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

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        learning_rate = CustomSchedule(ffdim, warmup_steps=1000)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, weight_decay=1e-4)
        model = EncoderOnly(num_layers=num_layers, d_model=ffdim, image_size=image_size, patch_size=16)
        img = tf.random.normal((4, 256, 256, 1))
        p = tf.random.normal((4, 1))
        model([img, p])
        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=[MeanSquaredError(), CosineSimilarity(axis=-2)],
        )

    model.fit(
        trainset,
        epochs=epochs,
        validation_data=valset,
        callbacks=[tensorboard, checkpoint]
    )

    model.save_weights(output_model_file)


class ExportWrapper(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])
    def __call__(self, shapes, periods):
        return self.model([shapes, periods])