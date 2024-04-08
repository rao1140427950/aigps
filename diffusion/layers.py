"""
credit to:
    https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion
"""

from keras import layers, activations
import tensorflow as tf


class PaddedConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.padding2d = layers.ZeroPadding2D(padding)
        self.conv2d = layers.Conv2D(filters, kernel_size, strides=strides)

    def call(self, inputs, *args, **kwargs):
        x = self.padding2d(inputs)
        return self.conv2d(x)

class PaddedConv2DNormActivation(layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, norm='gn', activation='swish', **kwargs):
        super(PaddedConv2DNormActivation, self).__init__(**kwargs)
        self.pconv = PaddedConv2D(filters, kernel_size, padding, strides)
        if norm == 'gn':
            self.norm = layers.GroupNormalization(epsilon=1e-5)
        else:
            self.norm = layers.GroupNormalization(epsilon=1e-5)

        self.act = layers.Activation(activation)

    def call(self, inputs, *args, **kwargs):
        x = self.pconv(inputs)
        x = self.norm(x)
        x = self.act(x)
        return x


class DenseNormActivation(layers.Layer):
    def __init__(self, units, norm='bn', activation='swish'):
        super(DenseNormActivation, self).__init__()
        self.dense = layers.Dense(units)
        if norm == 'bn':
            self.norm = layers.BatchNormalization()
        else:
            self.norm = layers.BatchNormalization()
        self.act = layers.Activation(activation)

    def call(self, inputs, *args, **kwargs):
        x = self.dense(inputs)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResBlock(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.residual_projection = None
        self.output_dim = output_dim
        self.entry_flow = [
            layers.GroupNormalization(epsilon=1e-5),
            layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]
        self.embedding_flow = [
            layers.Activation("swish"),
            layers.Dense(output_dim),
        ]
        self.exit_flow = [
            layers.GroupNormalization(epsilon=1e-5),
            layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]

    def build(self, input_shape):
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs, *args, **kwargs):
        inputs, embeddings = inputs
        x = inputs
        for layer in self.entry_flow:
            x = layer(x)
        for layer in self.embedding_flow:
            embeddings = layer(embeddings)
        x = x + embeddings[:, None, None, :]
        for layer in self.exit_flow:
            x = layer(x)
        return x + self.residual_projection(inputs)


class SpatialTransformer(layers.Layer):
    def __init__(self, num_heads, head_size, fully_connected=False, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.GroupNormalization(epsilon=1e-5)
        channels = num_heads * head_size
        if fully_connected:
            self.proj1 = layers.Dense(num_heads * head_size)
        else:
            self.proj1 = PaddedConv2D(num_heads * head_size, 1)
        self.transformer_block = BasicTransformerBlock(
            channels, num_heads, head_size
        )
        if fully_connected:
            self.proj2 = layers.Dense(channels)
        else:
            self.proj2 = PaddedConv2D(channels, 1)

    def call(self, inputs, *args, **kwargs):
        inputs, context = inputs
        _, h, w, c = inputs.shape
        x = self.norm(inputs)
        x = self.proj1(x)
        x = tf.reshape(x, (-1, h * w, c))
        x = self.transformer_block([x, context])
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj2(x) + inputs


class BasicTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(num_heads, head_size)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(num_heads, head_size)
        self.norm3 = layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = layers.Dense(dim)

    def call(self, inputs, *args, **kwargs):
        inputs, context = inputs
        x = self.attn1([self.norm1(inputs), None]) + inputs
        x = self.attn2([self.norm2(x), context]) + x
        return self.dense(self.geglu(self.norm3(x))) + x


def td_dot(a, b):
    aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = layers.Dot(axes=(2, 1))([aa, bb])
    return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))


class CrossAttention(layers.Layer):
    def __init__(self, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.to_q = layers.Dense(num_heads * head_size, use_bias=False)
        self.to_k = layers.Dense(num_heads * head_size, use_bias=False)
        self.to_v = layers.Dense(num_heads * head_size, use_bias=False)
        self.scale = head_size**-0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = layers.Dense(num_heads * head_size)

    def call(self, inputs, *args, **kwargs):
        inputs, context = inputs
        if context is None:
            context = inputs
        q, k, v = self.to_q(inputs), self.to_k(context), self.to_v(context)
        q = tf.reshape(
            q, (-1, inputs.shape[1], self.num_heads, self.head_size)
        )
        k = tf.reshape(
            k, (-1, context.shape[1], self.num_heads, self.head_size)
        )
        v = tf.reshape(
            v, (-1, context.shape[1], self.num_heads, self.head_size)
        )

        q = tf.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = tf.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = tf.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = activations.softmax(
            score
        )  # (bs, num_heads, time, time)
        attn = td_dot(weights, v)
        attn = tf.transpose(
            attn, (0, 2, 1, 3)
        )  # (bs, time, num_heads, head_size)
        out = tf.reshape(
            attn, (-1, inputs.shape[1], self.num_heads * self.head_size)
        )
        return self.out_proj(out)


class Upsample(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.ups = layers.UpSampling2D(2)
        self.conv = PaddedConv2D(channels, 3, padding=1)

    def call(self, inputs, *args, **kwargs):
        return self.conv(self.ups(inputs))


class GEGLU(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = layers.Dense(output_dim * 2)

    def call(self, inputs, *args, **kwargs):
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = activations.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate**2))
        )
        return x * 0.5 * gate * (1 + tanh_res)


class ResnetBlock(layers.Layer):
    def __init__(self, output_dim, kernel_size=3, padding=1, **kwargs):
        super().__init__(**kwargs)
        self.residual_projection = None
        self.output_dim = output_dim
        self.norm1 = layers.GroupNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(output_dim, kernel_size, padding=padding)
        self.norm2 = layers.GroupNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(output_dim, kernel_size, padding=padding)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(activations.swish(self.norm1(inputs)))
        x = self.conv2(activations.swish(self.norm2(x)))
        return x + self.residual_projection(inputs)


class AttentionBlock(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = layers.GroupNormalization(epsilon=1e-5)
        self.q = PaddedConv2D(output_dim, 1)
        self.k = PaddedConv2D(output_dim, 1)
        self.v = PaddedConv2D(output_dim, 1)
        self.proj_out = PaddedConv2D(output_dim, 1)

    def call(self, inputs, *args, **kwargs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        shape = tf.shape(q)
        h, w, c = shape[1], shape[2], shape[3]
        q = tf.reshape(q, (-1, h * w, c))  # b, hw, c
        k = tf.transpose(k, (0, 3, 1, 2))
        k = tf.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * 1 / tf.sqrt(tf.cast(c, tf.float32))
        y = activations.softmax(y)

        # Attend to values
        v = tf.transpose(v, (0, 3, 1, 2))
        v = tf.reshape(v, (-1, c, h * w))
        y = tf.transpose(y, (0, 2, 1))
        x = v @ y
        x = tf.transpose(x, (0, 2, 1))
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs