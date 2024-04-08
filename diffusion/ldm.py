"""
credit to:
    https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion
"""

import keras
from keras import layers
from diffusion.layers import PaddedConv2D, ResBlock, SpatialTransformer, Upsample
from diffusion.condition_encoder import get_condition_inputs


def latent_diffusion(
        image_size=32,
        image_channels=2,
        temb_ffdim=256,
        ffdims=(64, 128, 256),
        head_size=16,
        cond_type='c4shape_power',
):
    image = keras.layers.Input(
        (image_size, image_size, image_channels), name="image"
    )
    t_embed_input = keras.layers.Input((temb_ffdim,), name="timestep_embedding")

    cond_inputs, cond_latent = get_condition_inputs(cond_type)


    cond_latent = keras.layers.Dense(ffdims[2])(cond_latent)
    cond_latent = keras.layers.Activation("swish")(cond_latent)
    cond_latent = keras.layers.LayerNormalization()(cond_latent)
    cond_latent = keras.layers.Dense(ffdims[2])(cond_latent)
    cond_latent = keras.layers.Activation("swish")(cond_latent)
    cond_latent = keras.layers.LayerNormalization()(cond_latent)
    cond_latent = keras.layers.Dense(ffdims[2])(cond_latent)
    cond_latent = keras.layers.Activation("swish")(cond_latent)
    cond_latent = keras.layers.LayerNormalization()(cond_latent)
    cond_latent = keras.layers.Dense(ffdims[2])(cond_latent)

    t_emb = keras.layers.Dense(ffdims[2])(t_embed_input)
    t_emb = keras.layers.Activation("swish")(t_emb)
    t_emb = keras.layers.Dense(ffdims[2])(t_emb)

    # Downsampling flow

    outputs = []
    x = PaddedConv2D(ffdims[0], kernel_size=3, padding=1)(image)
    outputs.append(x)

    for _ in range(2):
        x = ResBlock(ffdims[0])([x, t_emb])
        x = SpatialTransformer(ffdims[0] // head_size, head_size, fully_connected=True)([x, cond_latent])
        outputs.append(x)
    x = PaddedConv2D(ffdims[0], 3, strides=2, padding=1)(x)  # Downsample 2x
    outputs.append(x)

    for _ in range(2):
        x = ResBlock(ffdims[1])([x, t_emb])
        x = SpatialTransformer(ffdims[1] // head_size, head_size, fully_connected=True)([x, cond_latent])
        outputs.append(x)
    x = PaddedConv2D(ffdims[1], 3, strides=2, padding=1)(x)  # Downsample 2x
    outputs.append(x)

    for _ in range(2):
        x = ResBlock(ffdims[2])([x, t_emb])
        x = SpatialTransformer(ffdims[2] // head_size, head_size, fully_connected=True)([x, cond_latent])
        outputs.append(x)
    x = PaddedConv2D(ffdims[2], 3, strides=2, padding=1)(x)  # Downsample 2x
    outputs.append(x)

    for _ in range(2):
        x = ResBlock(ffdims[2])([x, t_emb])
        outputs.append(x)

    # Middle flow

    x = ResBlock(ffdims[2])([x, t_emb])
    x = SpatialTransformer(ffdims[2] // head_size, head_size, fully_connected=True)([x, cond_latent])
    x = ResBlock(ffdims[2])([x, t_emb])

    # Upsampling flow

    for _ in range(3):
        x = keras.layers.Concatenate()([x, outputs.pop()])
        x = ResBlock(ffdims[2])([x, t_emb])
    x = Upsample(ffdims[2])(x)

    for _ in range(3):
        x = keras.layers.Concatenate()([x, outputs.pop()])
        x = ResBlock(ffdims[2])([x, t_emb])
        x = SpatialTransformer(ffdims[2] // head_size, head_size, fully_connected=True)([x, cond_latent])
    x = Upsample(ffdims[2])(x)

    for _ in range(3):
        x = keras.layers.Concatenate()([x, outputs.pop()])
        x = ResBlock(ffdims[1])([x, t_emb])
        x = SpatialTransformer(ffdims[1] // head_size, head_size, fully_connected=True)([x, cond_latent])
    x = Upsample(ffdims[1])(x)

    for _ in range(3):
        x = keras.layers.Concatenate()([x, outputs.pop()])
        x = ResBlock(ffdims[0])([x, t_emb])
        x = SpatialTransformer(ffdims[0] // head_size, head_size, fully_connected=True)([x, cond_latent])

    # Exit flow

    x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
    x = keras.layers.Activation("swish")(x)
    output = PaddedConv2D(image_channels, kernel_size=3, padding=1)(x)

    return keras.Model([image, t_embed_input] + cond_inputs, output, name='ldm')


if __name__ == '__main__':
    net = latent_diffusion()
    pass