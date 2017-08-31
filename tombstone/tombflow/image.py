# encoding: UTF-8

import tensorflow as tf


def rgb_to_bgr(image, scope=None):
    with tf.name_scope(scope or 'rgb_to_bgr'):
        r, g, b = tf.split(image, num_or_size_splits=3, axis=2)
        bgr = tf.concat([b, g, r], axis=2)
    return bgr
