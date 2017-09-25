# encoding: UTF-8

import tensorflow as tf
from tensorflow.contrib.data import TextLineDataset
from tombstone.tombflow.image import rgb_to_bgr, resize as tf_resize
from enum import Enum
import os
from functools import partial


class SequentialImageReader(object):
    def __init__(self, sequence, data_dir=None, name_scope=None, mean=None, dtype=tf.float32, use_bgr=False, modifier=None, expand_dims_at=None):
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
            
            if expand_dims_at is not None:
                self.image_dataset = self.image_dataset.map(lambda x: tf.expand_dims(x, axis=expand_dims_at))

            self.path_iterator, self.path_initializer, self.path = self.build_iterator(self.path_dataset)
            self.image_iterator, self.image_initializer, self.image = self.build_iterator(self.image_dataset)
            self.initializers = [self.path_initializer, self.image_initializer]
            self.total = tf.shape(tf.string_split(tf.expand_dims(tf.read_file(sequence), axis=0), delimiter=b'\n'))[-1]

    @staticmethod
    def build_iterator(dataset):
        iterator = dataset.make_initializable_iterator()
        initializer = iterator.initializer
        value = iterator.get_next()
        return iterator, initializer, value


class VOC2012Subsets(Enum):
    seg_val = 'Segmentation/val.txt'
    seg_train = 'Segmentation/train.txt'
    seg_trainval = 'Segmentation/trainval.txt'


def tf_path_join(*args, ext=None):
    path = tf.string_join(args, separator=os.path.sep)
    if ext is not None:
        path = tf.string_join((path, ext), separator='.')
    return path


class VOC2012Dataset:
    def __init__(self, voc_root, subset, name_scope=None, mean=None,
                 dtype=tf.float32, use_bgr=True, expand_dims_at=None, enable_label=None,
                 batch_size=1, resize=None):
        assert batch_size is 1 or resize is not None
        with tf.name_scope(name_scope or 'VOC2012Dateset'):
            if isinstance(subset, VOC2012Subsets):
                ss = os.path.join('ImageSets', subset.value)
            else:
                ss = str(subset)
            self.serials = TextLineDataset(os.path.join(voc_root, ss))

            self.image_dataset = (self.serials.map(lambda x: tf_path_join(voc_root, 'JPEGImages', x, ext='jpg'))
                                              .map(tf.read_file)
                                              .map(tf.image.decode_jpeg))
            if resize is not None:
                self.image_dataset = self.image_dataset.map(lambda x: tf_resize(x, resize))

            self.image_dataset = self.image_dataset.batch(batch_size)

            enable_label = enable_label if enable_label is not None else 'test' not in ss
            if enable_label:
                self.label_dataset = (self.serials
                                          .map(lambda x: tf_path_join(voc_root, 'SegmentationClass', x, ext='png'))
                                          .map(tf.read_file)
                                          .map(tf.image.decode_png))
                if resize is not None:
                    self.label_dataset = self.label_dataset.map(lambda x: tf_resize(x, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
                    self.label_dataset = self.label_dataset.batch(batch_size)

            if use_bgr:
                self.image_dataset = self.image_dataset.map(rgb_to_bgr)

            if dtype is not None:
                self.image_dataset = self.image_dataset.map(lambda x: tf.cast(x, dtype=dtype))

            if mean is not None:
                self.image_dataset = self.image_dataset.map(lambda x: x - mean)

            if expand_dims_at is not None:
                self.image_dataset = self.image_dataset.map(lambda x: tf.expand_dims(x, axis=expand_dims_at))

            self.serial_iterator, self.serial_initializer, self.serial = SequentialImageReader.build_iterator(self.serials)
            self.image_iterator, self.image_initializer, self.image = SequentialImageReader.build_iterator(self.image_dataset)
            self.initializers = [self.serial_initializer, self.image_initializer]
            if enable_label:
                self.label_iterator, self.label_initializer, self.label = SequentialImageReader.build_iterator(self.label_dataset)
                self.initializers.append(self.label_initializer)
            self.total = tf.shape(tf.string_split(tf.expand_dims(tf.read_file(ss), axis=0), delimiter=b'\n'))[-1]
