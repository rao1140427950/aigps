import tensorflow as tf
import numpy as np
import scipy.io as sio
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


class PatternsDataset:

    def __init__(
            self,
            mat_path=None,
            tfrecord_path=None,
            key='shapes',
            image_size=128,
            shuffle=True,
            random_roll=False,
            random_invert=False,
            random_mask=False,
    ):
        super(PatternsDataset, self).__init__()
        self.mat_path = mat_path
        self.tfrecord_path = tfrecord_path
        self.matkey = key
        self.image_size = image_size
        self.shuffle = shuffle
        self.random_roll = random_roll
        self.random_invert = random_invert

        self.random_mask = random_mask
        if random_mask:
            mask = sio.loadmat('../random_masks_50k.mat')['mask'].astype(np.float32)
            self.mask = tf.data.Dataset.from_tensor_slices(mask).repeat(5).shuffle(10000)
        else:
            self.mask = None

    @staticmethod
    def _float_list_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def train_preprocessing(self, img):
        img = tf.cast(img, dtype=tf.float32)
        img = tf.image.resize(img, size=(self.image_size, self.image_size),
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)

        if self.random_roll:
            shift = tf.random.uniform(shape=(2,), minval=0, maxval=self.image_size, dtype=tf.int32)
            img = tf.roll(img, shift, axis=(0, 1))

        def fcn1():
            return img
        def fcn2():
            return 1. - img

        if self.random_invert:
            img = tf.cond(tf.greater(tf.random.uniform([], 0., 1.), 0.5), true_fn=fcn1, false_fn=fcn2)

        img = img * 2. - 1.
        return img

    def generate_tfrecords_from_mat(self, save_path=None, label=0):
        # 0 for C4
        if save_path is None:
            save_path = self.tfrecord_path
        shapes = sio.loadmat(self.mat_path)[self.matkey]
        length = shapes.shape[0]

        idxs = np.arange(length)
        np.random.shuffle(idxs)

        writer = tf.io.TFRecordWriter(save_path)
        for p in range(length):
            png = tf.image.encode_png(1 - shapes[idxs[p], :, :, np.newaxis])
            feature = {'pattern': self._bytes_feature(png), 'label': self._int64_feature(label)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

    def generate_dataset_from_tfrecords(self, batch_size=16, return_label=True):

        feature_description = {'pattern': tf.io.FixedLenFeature([], tf.string),
                               'label': tf.io.FixedLenFeature([], tf.int64)}
        def _parse_example_function(example_proto):
            example = tf.io.parse_single_example(example_proto, feature_description)
            pattern = example['pattern']
            pattern = tf.image.decode_png(pattern, channels=1)
            pattern = self.train_preprocessing(pattern)
            label = example['label'] + 1

            def fcn1():
                return tf.cast(label, tf.int64)
            def fcn2():
                return tf.cast(0, tf.int64)
            label = tf.cond(tf.greater(tf.random.uniform([], 0., 1.), 0.3), true_fn=fcn1, false_fn=fcn2)

            if return_label:
                return pattern, label
            else:
                return pattern

        def _mask_map_fcn(mask):
            rate = 0.5

            def fcn1():
                return mask
            def fcn2():
                return tf.ones_like(mask)

            return tf.cond(tf.greater(tf.random.normal([], 0., 1.), rate), fcn1, fcn2)

        dataset = tf.data.TFRecordDataset(self.tfrecord_path, num_parallel_reads=len(self.tfrecord_path))
        if self.shuffle:
            dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
        dataset = dataset.map(_parse_example_function, num_parallel_calls=4)
        if self.random_mask:
            dataset = tf.data.Dataset.zip((dataset, self.mask.map(_mask_map_fcn)))
        return dataset.batch(batch_size, drop_remainder=True).prefetch(32)


def data_map_fcn(data, mask):
    step = 5
    nsteps = 15
    idx = 100 - np.abs(np.arange(201) - 100)

    fft = tf.signal.fft(tf.cast(data, tf.complex64))
    features = [data[tf.newaxis, ...]]
    for n in reversed(range(nsteps)):
        pos = n * step
        temp = tf.where(idx > pos, tf.cast(0. + 0.j, tf.complex64), fft)
        ifft = tf.signal.ifft(temp)
        features.append(tf.math.real(ifft[tf.newaxis, ...]))
    features = tf.concat(features, axis=0)
    features = tf.transpose(features)
    return features, mask


def mask_map_fcn(data, mask):
    rate = 0.5

    def fcn1():
        return mask
    def fcn2():
        return tf.ones_like(mask)

    return data, tf.cond(tf.greater(tf.random.uniform([], 0., 1.), rate), fcn1, fcn2)


def discrete_points_dataset():
    mat = sio.loadmat('./runtime_data/random_points_20k.mat')
    data = mat['data']
    mask = mat['mask'].astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((data, mask))
    dataset = dataset.map(data_map_fcn)
    return dataset


def random_filters_dataset():
    mat = sio.loadmat('./runtime_data/random_filters_20k.mat')
    data = mat['data']
    mask = mat['mask'].astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((data, mask))
    dataset = dataset.map(mask_map_fcn).map(data_map_fcn)
    return dataset


def shapes_tpower_dataset():
    data = sio.loadmat('./runtime_data/shapes_tpower_50k.mat')['data']
    mask = sio.loadmat('./runtime_data/random_masks_50k.mat')['mask'].astype(np.float32)

    data = tf.data.Dataset.from_tensor_slices(data).shuffle(50000)
    mask = tf.data.Dataset.from_tensor_slices(mask).shuffle(50000)
    dataset = tf.data.Dataset.zip((data, mask))
    dataset = dataset.map(mask_map_fcn).map(data_map_fcn)
    return dataset

def phase_mask_map_fcn(data, mask):
    rate = 0.2

    def fcn1():
        return mask
    def fcn2():
        return tf.ones_like(mask)

    # data = data / (np.pi * 2.)
    cosdata = tf.math.cos(data)
    sindata = tf.math.sin(data)
    data = tf.stack([cosdata, sindata], axis=1)

    return data, tf.cond(tf.greater(tf.random.uniform([], 0., 1.), rate), fcn1, fcn2)


def shapes_tphase_dataset(batch_size=16):
    data = sio.loadmat('./runtime_data/shapes_tphase_80k.mat')['data']
    mask1 = sio.loadmat('./runtime_data/random_masks_50k.mat')['mask'].astype(np.float32)
    mask2 = sio.loadmat('./runtime_data/tphase_masks_10k.mat')['masks'].astype(np.float32)
    mask = np.concatenate([mask1, mask2], axis=0)

    data = tf.data.Dataset.from_tensor_slices(data).shuffle(50000)
    mask = tf.data.Dataset.from_tensor_slices(mask).shuffle(50000)
    dataset = tf.data.Dataset.zip((data, mask))
    dataset = dataset.map(phase_mask_map_fcn)
    return dataset.batch(batch_size).prefetch(16)


def get_masked_tpower_dataset(batch_size=16):
    dataset1_1 = discrete_points_dataset()
    dataset1_2 = random_filters_dataset()

    dataset1 = dataset1_2.concatenate(dataset1_1).shuffle(40000)
    dataset2 = shapes_tpower_dataset().shuffle(10000)

    choice_dataset = tf.data.Dataset.range(2).repeat(40000)
    dataset = tf.data.Dataset.choose_from_datasets([dataset1, dataset2], choice_dataset)

    return dataset.batch(batch_size).prefetch(8)


def _map_grouped_dataset(data, mask, order=4):
    ref = ref_out = tf.stack([data[0, ...]] * 4, axis=0)
    ndata = tf.where(mask[..., tf.newaxis] > 0.5, data, 0.)
    ref = tf.where(mask[..., tf.newaxis] > 0.5, ref, 0.)
    ndata = tf.math.l2_normalize(ndata, axis=-2)
    ref = tf.math.l2_normalize(ref, axis=-2)
    sims = tf.math.reduce_sum(ndata * ref, axis=-2)
    sim = sims[:, -order] ** 5

    return data, mask, ref_out, sim


def get_paired_tpower_dataset(batch_size=16):
    dataset1_1 = discrete_points_dataset().shuffle(10000)
    dataset1_2 = random_filters_dataset().shuffle(10000)

    dataset2 = shapes_tpower_dataset().shuffle(20000)

    choice_dataset = tf.data.Dataset.from_tensor_slices(tf.cast([0, 0, 1, 2], tf.int64)).repeat(20000)
    dataset_grouped = tf.data.Dataset.choose_from_datasets([dataset2, dataset1_1, dataset1_2], choice_dataset)

    dataset = dataset_grouped.batch(4).map(_map_grouped_dataset)
    dataset = dataset.unbatch().shuffle(10000).batch(batch_size).prefetch(8)
    return dataset