# encoding: UTF-8

import tensorflow as tf


def rgb_to_bgr(image, scope=None):
    with tf.name_scope(scope or 'rgb_to_bgr'):
        bgr = tf.reverse(image, axis=[-1])
    return bgr


def resize(image, new_shape, method=tf.image.ResizeMethod.BILINEAR):
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize_images(image, new_shape, method=method)
    return image[0]

