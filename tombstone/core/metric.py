# encoding: UTF-8

"""
Confusing matrix based metrics borrowed from
https://github.com/shelhamer/fcn.berkeleyvision.org
"""
    
from __future__ import division
import numpy as np
import sys


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.loss = 0
        self.__samples__ = 0

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.num_classes)
        return np.bincount(self.num_classes * a[k].astype(int) + b[k], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def update_hist(self, prediction: np.ndarray, label: np.ndarray, loss: float=None):
        self.hist += self.fast_hist(label.flatten(), prediction.flatten())
        if loss is not None:
            self.loss += loss
            self.__samples__ += 1

    @property
    def mean_loss(self):
        return self.loss / self.__samples__

    @property
    def overall_accuracy(self):
        return np.diag(self.hist).sum() / self.hist.sum()

    @property
    def perclass_accuracy(self):
        return np.diag(self.hist) / self.hist.sum(1)

    @property
    def mean_accuracy(self):
        return np.nanmean(self.perclass_accuracy)

    @property
    def iou(self):
        diag = np.diag(self.hist)
        return diag / (self.hist.sum(1) + self.hist.sum(0) - diag)

    @property
    def mean_iou(self):
        return np.nanmean(self.iou())

    @property
    def fwavacc(self):
        return self.hist.sum(1) / self.hist.sum()

    def report(self, stream=sys.stdout, indent=0):
        assert type(indent) is int
        indent = ' ' * indent
        reports = [
            '{}Mean Accuracy: {:.2%}'.format(indent, self.mean_accuracy),
            '{}Mean IoU:      {:.2%}'.format(indent, self.iou),
            '{}FWAV Accuracy: {:.2%}'.format(indent, self.fwavacc),
            '{}Overall Accuracy: {:.2%}'.format(indent, self.overall_accuracy)
        ]
        if self.__samples__ > 0:
            reports.append('{}Mean Loss:     {:.6f}'.format(indent, self.mean_loss))
        stream.writelines(reports)
        stream.flush()
