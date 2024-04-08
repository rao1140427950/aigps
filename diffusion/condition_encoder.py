import tensorflow as tf
from keras import layers


def c4shape_condition(
        shape_types=2,
        cond_ffdim=128,
):
    shape_condition_input = layers.Input(shape=(), dtype=tf.int64, name="shape_control")

    sc = layers.Flatten()(shape_condition_input)
    sc = layers.Embedding(shape_types + 1, cond_ffdim, input_length=1)(sc)
    sc = layers.Reshape((1, cond_ffdim))(sc)

    return [shape_condition_input], sc


def power_condition(
        cond_length=201,
        cond_ffdim=128,
):
    cond_inputs = layers.Input((cond_length, cond_ffdim), name="power_cond")

    return [cond_inputs], cond_inputs


def polarized_power_condition(
        cond_length=201,
        cond_ffdim=128,
        combine_type='hconcat',
):
    h_cond_inputs = layers.Input((cond_length, cond_ffdim), name="h_power_cond")
    v_cond_inputs = layers.Input((cond_length, cond_ffdim), name="v_power_cond")

    if combine_type == 'add':
        cond = h_cond_inputs + v_cond_inputs
    elif combine_type == 'vconcat':
        cond = tf.concat([h_cond_inputs, v_cond_inputs], axis=-2)
    elif combine_type == 'hconcat':
        cond = tf.concat([h_cond_inputs, v_cond_inputs], axis=-1)
    else:
        raise ValueError('Unknown input for `combine_type`.')

    return [h_cond_inputs, v_cond_inputs], cond


def c4shape_power_condition(
        shape_types=2,
        cond_length=201,
        cond_ffdim=128,
        combine_type='vconcat',
):

    shape_condition_input = layers.Input(shape=(), dtype=tf.int64, name="shape_control")
    cond_inputs = layers.Input((cond_length, cond_ffdim), name="power_cond")

    sc = layers.Flatten()(shape_condition_input)
    sc = layers.Embedding(shape_types + 1, cond_ffdim, input_length=1)(sc)
    sc = layers.Reshape((1, cond_ffdim))(sc)

    if combine_type == 'add':
        cond = cond_inputs + sc
    elif combine_type == 'vconcat':
        cond = tf.concat([cond_inputs, sc], axis=-2)
    else:
        raise ValueError('Unknown input for `combine_type`.')

    return [shape_condition_input, cond_inputs], cond


def c4shape_phase_condition(
        shape_types=2,
        cond_length=201,
        cond_ffdim=128,
        combine_type='vconcat',
):

    return c4shape_power_condition(
        shape_types,
        cond_length,
        cond_ffdim,
        combine_type,
    )


def phase_condition(
        cond_length=201,
        cond_ffdim=128,
):
    return power_condition(
        cond_length,
        cond_ffdim,
    )


def polarized_phase_condition(
        cond_length=201,
        cond_ffdim=128,
        combine_type='hconcat',
):
    return polarized_power_condition(
        cond_length,
        cond_ffdim,
        combine_type,
    )


def get_condition_inputs(
        type_='c4shape_power',
):
    if type_ == 'c4shape_power':
        return c4shape_power_condition()
    elif type_ == 'power':
        return power_condition()
    elif type_ == 'polarized_power':
        return polarized_power_condition()
    elif type_ == 'c4shape_phase':
        return c4shape_power_condition()
    elif type_ == 'phase':
        return power_condition()
    elif type_ == 'polarized_phase':
        return polarized_power_condition()
    elif type_ == 'c4shape':
        return c4shape_condition()
    else:
        raise ValueError('Unknown input for `type_`.')