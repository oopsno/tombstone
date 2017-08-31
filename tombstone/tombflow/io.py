# encoding: UTF-8

import tensorflow as tf
from tensorflow.contrib.data import TextLineDataset
from tombstone.tombflow.image import rgb_to_bgr


class SequentialImageReader(object):
    def __init__(self, sequence, data_dir=None, name_scope=None, mean=None, dtype=tf.float32, use_bgr=False, modifier=None):
        with tf.name_scope(name_scope or 'SequentialImageReader'):
            self.path_dataset = TextLineDataset(sequence)
            if modifier is not None:
                self.path_dataset = self.path_dataset.map(modifier)
            if data_dir is not None:
                data_dir = tf.constant([data_dir], dtype=tf.string, name='data_dir')
                prefix = TextLineDataset.from_tensors(data_dir).repeat()
                self.path_dataset = (TextLineDataset.zip((prefix, self.path_dataset))
                                                    .map(lambda pre, filename: pre[0] + filename))

            self.image_dataset = (self.path_dataset
                                      .map(tf.read_file)
                                      .map(tf.image.decode_image))

            if use_bgr:
                self.image_dataset = self.image_dataset.map(rgb_to_bgr)

            if dtype is not None:
                self.image_dataset = self.image_dataset.map(lambda x: tf.cast(x, dtype=dtype))

            if mean is not None:
                self.image_dataset = self.image_dataset.map(lambda x: x - mean)

            self.path_iterator, self.path_initializer, self.path = self.build_iterator(self.path_dataset)
            self.image_iterator, self.image_initializer, self.image = self.build_iterator(self.image_dataset)
            self.initializers = [self.path_initializer, self.image_initializer]

    @staticmethod
    def build_iterator(dataset):
        iterator = dataset.make_initializable_iterator()
        initializer = iterator.initializer
        value = iterator.get_next()
        return iterator, initializer, value

