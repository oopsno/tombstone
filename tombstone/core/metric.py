# encoding: UTF-8

"""
Confusing matrix based metrics borrowed from
https://github.com/shelhamer/fcn.berkeleyvision.org
"""
    
from __future__ import division
import numpy as np


class Metric:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.loss = 0
        self.__samples__ = 0

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.num_classes)
        return np.bincount(self.num_classes * a[k].astype(int) + b[k], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def update_hist(self, prediction: np.ndarray, label: np.ndarray, loss: float):
        self.hist += self.fast_hist(label.flatten(), prediction.flatten())
        self.loss += loss
        self.__samples__ += 1

    def mean_loss(self):
        return self.loss / self.__samples__

    def overall_accuracy(self):
        return np.diag(self.hist).sum() / self.hist.sum()

    def mean_accuracy(self):
        return np.diag(self.hist) / self.hist.sum(1)

    def mean_iou(self):
        diag = np.diag(self.hist)
        return diag / (self.hist.sum(1) + self.hist.sum(0) - diag)

    def fwavacc(self):
        return self.hist.sum(1) / self.hist.sum()