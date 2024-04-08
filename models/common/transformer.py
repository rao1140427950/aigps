import numpy as np
import tensorflow as tf
from keras import layers, activations


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)

    angle_rates = 1 / (10000 ** depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(layers.Layer):
    def __init__(self, length, depth, trainable_embedding=False, rescale=False):
        super(PositionalEmbedding, self).__init__()
        self.length = length
        self.depth = depth
        self.rescale = rescale
        if trainable_embedding:
            self.position_emb = self.add_weight(shape=(1, length, depth), dtype=tf.float32, name='position_embd',
                                                initializer=tf.keras.initializers.random_normal(), trainable=True)
        else:
            self.position_emb = positional_encoding(length, depth)[tf.newaxis, ...]

    def call(self, inputs, *args, **kwargs):
        if self.rescale:
            inputs *= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        length = tf.shape(inputs)[1]
        return inputs + self.position_emb[:, :length, :]


class PatchEmbedding(layers.Layer):
    def __init__(self, image_size=256, patch_size=16, emb_size=256, cls_token=False, **kwargs):
        super().__init__(**kwargs)
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.emb_size = emb_size

        self.proj = layers.Conv2D(emb_size, kernel_size=patch_size, strides=patch_size)
        self.reshape = layers.Reshape((-1, emb_size))
        if cls_token:
            self.cls_token = self.add_weight(shape=(1, 1, emb_size), dtype=tf.float32, name='class_token',
                                             initializer=tf.keras.initializers.random_normal(), trainable=True)
        else:
            self.cls_token = None

    def build(self, input_shape):
        b, h, w, c = input_shape
        assert h == self.image_size
        assert w == self.image_size
        super(PatchEmbedding, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.proj(inputs)  # (b, h // p, w // p, emb)
        x = self.reshape(x)  # (b, nh * nw, emb)

        if self.cls_token is not None:
            bsize = tf.shape(inputs)[0]
            cls_token = tf.tile(self.cls_token, (bsize, 1, 1))
            x = tf.concat([cls_token, x], axis=1)  # (b, nh * nw + 1, emb)

        return x


class FeatureEmbedding(layers.Layer):
    """
    input: (bsize, seq_length, 2). Each element is normalized to [-1, 1]
    """
    def __init__(self, emb_size=256, **kwargs):
        super(FeatureEmbedding, self).__init__(**kwargs)
        self.proj = layers.Dense(emb_size)
        self.emb_size = emb_size

    def call(self, inputs, *args, **kwargs):
        x = self.proj(inputs)
        return x


class EncoderPreprocessing(layers.Layer):

    def __init__(
            self,
            image_size=256,
            patch_size=16,
            emb_size=256,
            period_token=True,
            trainable_positional_embedding=False,
            rescale=False,
    ):
        super(EncoderPreprocessing, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, emb_size)
        npos = (image_size // patch_size) ** 2
        self.period_token = period_token
        if period_token:
            npos += 1
            self.period_proj = layers.Dense(emb_size)
        self.pos_embedding = PositionalEmbedding(npos, emb_size, trainable_positional_embedding, rescale)
        self.npos = npos

    def call(self, inputs, *args, **kwargs):
        if self.period_token:
            images, periods = inputs  # (bsize, imgsize, imgsize, 1), (bsize, 1)
            periods = self.period_proj(periods)  # (bsize, emb)
            images = self.patch_embedding(images)  # (b, nh * nw, emb)
            context = tf.concat([images, periods[:, tf.newaxis, :]], axis=1)
        else:
            context = self.patch_embedding(inputs)

        return self.pos_embedding(context)


class FeedForward(layers.Layer):
    def __init__(self, in_dim, expansion=4, dropout=0.1, activation=activations.relu, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(in_dim * expansion)
        self.act = activation
        self.dense2 = layers.Dense(in_dim)
        self.dropout = layers.Dropout(dropout)
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self.dense1(inputs)
        x = self.act(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.add([x, inputs])
        x = self.layer_norm(x)
        return x


class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()
        self.last_attn_scores = None


class CrossAttention(BaseAttention):
    def call(self, inputs, *args, **kwargs):
        x, context = inputs
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x, *args, **kwargs):
        attn_output, attn_scores = self.mha(
            query=x,
            value=x,
            key=x,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x, *args, **kwargs):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class EncoderLayer(layers.Layer):
    """
    inputs: (bsize, seq_length, ffdim)
    outputs: (bsize, seq_length, ffdim)
    """

    def __init__(self, num_heads, emb_size, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = GlobalSelfAttention(num_heads=num_heads, key_dim=emb_size, dropout=dropout)
        self.ffn = FeedForward(emb_size, dropout=dropout)
        self.last_attn_scores = None

    def call(self, inputs, *args, **kwargs):
        x = self.self_attention(inputs)
        x = self.ffn(x)
        self.last_attn_scores = self.self_attention.last_attn_scores
        return x


class DecoderLayer(layers.Layer):
    """
    inputs: x: (bsize, seq1, ffdim)  context: (bsize, seq2, ffdim)
    outputs: (bsize, seq1, ffdim)
    """

    def __init__(self, num_heads, emb_size, dropout=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=emb_size, dropout=dropout)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=emb_size, dropout=dropout)
        self.ffn = FeedForward(emb_size, dropout=dropout)
        self.last_attn_scores = None

    def call(self, inputs, *args, **kwargs):
        x, context = inputs
        x = self.causal_self_attention(x)
        x = self.cross_attention([x, context])
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Encoder(layers.Layer):

    def __init__(self, num_layers, num_heads, emb_size, dropout=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.enc_layers = [EncoderLayer(num_heads, emb_size, dropout) for _ in range(num_layers)]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.enc_layers:
            x = layer(x)
        return x


class Decoder(layers.Layer):
    def __init__(self, num_layers, num_heads, emb_size, dropout=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dec_layers = [DecoderLayer(num_heads, emb_size, dropout) for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, inputs, *args, **kwargs):
        x, context = inputs
        for layer in self.dec_layers:
            x = layer([x, context])
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x


