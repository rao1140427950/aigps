import keras
import tensorflow as tf
from keras import layers
import numpy as np
from models.common.dataset import get_masked_tpower_dataset


class SequenceEncoder(layers.Layer):
    def __init__(
            self,
            input_ffdim=16,
            input_length=201,
            ffdim=64,
            **kwargs,
    ):
        super(SequenceEncoder, self).__init__(**kwargs)
        self.ffdim = ffdim
        self.input_ffdim = input_ffdim
        self.input_length = input_length
        self.mask_token = self.add_weight(shape=(1, 1, input_ffdim), dtype=tf.float32,
                                          initializer=tf.initializers.Constant(-1), trainable=True, name='mask_token')
        self.projection = layers.Dense(ffdim)
        self.position_embedding = layers.Embedding(input_dim=input_length, output_dim=ffdim)

    def call(self, inputs, *args, **kwargs):
        data, mask = inputs
        data_masked = tf.where(mask[..., tf.newaxis] > 0.5, data, self.mask_token)

        batch_size = tf.shape(data)[0]
        positions = tf.range(start=0, limit=self.input_length, delta=1, dtype=tf.int32)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(pos_embeddings, [batch_size, 1, 1])  # (bsize, nfeatures, projection_dim)

        masked_embeddings = self.projection(data_masked) + pos_embeddings  # (bsize, nfeatures, projection_dim)

        return masked_embeddings


class MLP(layers.Layer):

    def __init__(self, hidden_units,  dropout_rate, activation='gelu', **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.activation = 'gelu'

        self.dense_layers = []
        self.dropout_layers = []
        for units in hidden_units:
            self.dense_layers.append(layers.Dense(units, activation=activation))
            self.dropout_layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x)
        return x


class Transformer(layers.Layer):

    def __init__(self, num_heads, key_dim, mlp_units, eps=1e-6, dropout=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mlp_units = mlp_units
        self.eps = eps
        self.dropout = dropout
        self.ln1 = layers.LayerNormalization(epsilon=eps)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.add1 = layers.Add()
        self.ln2 = layers.LayerNormalization(epsilon=eps)
        self.mlp = MLP(mlp_units, dropout)
        self.add2 = layers.Add()

    def call(self, inputs, *args, **kwargs):
        x = skip = inputs
        x = self.ln1(x)
        att = self.mha(x, x)
        x = skip = self.add1([att, skip])

        x = self.ln2(x)
        x = self.mlp(x)
        x = self.add2([x, skip])
        return x


class Encoder(keras.Model):

    def __init__(
            self,
            num_heads=4,
            num_layers=4,
            ffdim=64,
            mlp_units=(128, 64),
            dropout=0.1,
            input_ffdim=16,
            input_length=201,
            **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)
        self.input_encoder = SequenceEncoder(input_ffdim=input_ffdim, input_length=input_length, ffdim=ffdim)
        self.transformer_layers = []
        for _ in range(num_layers):
            self.transformer_layers.append(Transformer(num_heads=num_heads, key_dim=ffdim, mlp_units=mlp_units, dropout=dropout))
        self.ln_layer = layers.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        data, mask = inputs
        x = self.input_encoder([data, mask])
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.ln_layer(x)
        return x


class Decoder(keras.Model):

    def __init__(
            self,
            num_heads=4,
            num_layers=2,
            ffdim=64,
            mlp_units=(128, 64),
            dropout=0.1,
            output_ffdim=16,
            **kwargs
    ):
        super(Decoder, self).__init__(**kwargs)
        self.input_dense = layers.Dense(ffdim)
        self.transformer_layers = []
        for _ in range(num_layers):
            self.transformer_layers.append(Transformer(num_heads=num_heads, key_dim=ffdim, mlp_units=mlp_units, dropout=dropout))
        self.ln_layer = layers.LayerNormalization()
        self.output_dense = layers.Dense(output_ffdim)

    def call(self, inputs, training=None, mask=None):
        x = self.input_dense(inputs)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.ln_layer(x)
        x = self.output_dense(x)
        return x


class MAETrainer(keras.Model):

    def __init__(
            self,
            encoder,
            decoder,
            **kwargs,
    ):
        super(MAETrainer, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder


    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def train_step(self, inputs):
        data, mask = inputs

        with tf.GradientTape() as tape:
            pred = self.call(inputs, training=True)
            loss = self.loss(data, pred)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return {"loss": loss}

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


def train():
    batch_size = 64 * 8
    epochs = 160
    model_name = 'tpower_mae_l.h5'
    work_dir = './checkpoints/tpower_mae/'
    output_model_file = work_dir + model_name
    checkpoint_path = work_dir + 'checkpoint-' + model_name

    dataset = get_masked_tpower_dataset(batch_size)

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
            num_layers=8,
            ffdim=128,
            mlp_units=(256, 128),
            dropout=0.1,
            input_ffdim=16,
            input_length=201,
        )
        decoder = Decoder(
            num_heads=4,
            num_layers=4,
            ffdim=64,
            mlp_units=(128, 64),
            dropout=0.1,
            output_ffdim=16,
        )

        model = MAETrainer(encoder, decoder)
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
        step = 5
        nsteps = 15
        idx = 100 - np.abs(np.arange(201) - 100)

        fft = tf.signal.fft(tf.cast(data, tf.complex64))
        features = [data[:, tf.newaxis, ...]]
        for n in reversed(range(nsteps)):
            pos = n * step
            temp = tf.where(idx[tf.newaxis, ...] > pos, tf.cast(0. + 0.j, tf.complex64), fft)
            ifft = tf.signal.ifft(temp)
            features.append(tf.math.real(ifft[:, tf.newaxis, ...]))
        features = tf.concat(features, axis=1)
        features = tf.transpose(features, [0, 2, 1])
        return features

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
        return self.decoder(latents)

