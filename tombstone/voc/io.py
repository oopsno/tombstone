# encoding: UTF-8

import numpy as np

from tombstone.voc.colormap import colormap

import numpy as np
import png


class PNGWriter:
    def write(self, path, prediction):
        with open(path, 'wb') as dest:
            writer = png.Writer(prediction.shape[1], prediction.shape[0], palette=colormap)
            writer.write(dest, prediction)


