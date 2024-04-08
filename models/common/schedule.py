import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model=10000, warmup_steps=2000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.minimum(tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2), 1e-4)

  def get_config(self):
    return {'d_model': self.d_model, 'warmup_steps': self.warmup_steps}
