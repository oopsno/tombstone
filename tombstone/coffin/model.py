# encoding: UTF-8

"""
A wrapper struggling to modernize regular caffe models
"""

import caffe


class Model:
    def __init__(self, prototxt, weights, phase):
        self.net = caffe.Net(prototxt, weights, phase)
        self.__ready__ = False

    @staticmethod
    def __align_shape__(shape):
        return [1] * (4 - len(shape)) + list(shape)

    def feed(self, **kwargs):
        for blob_name, value in kwargs.items():
            self.net.blobs[blob_name].reshape(*self.__align_shape__(value.shape))
            self.net.blobs[blob_name].data[...] = value
        self.__ready__ = False

    def forward(self):
        if not self.__ready__:
            self.net.forward()
            self.__ready__ = True

    def __getitem__(self, blob_name):
        self.forward()
        return self.net.blobs[blob_name].data
