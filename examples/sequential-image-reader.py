from tombstone.tombflow.io import SequentialImageReader
import tensorflow as tf

import numpy as np

sess = tf.InteractiveSession()
sir = SequentialImageReader('VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt',
                            data_dir='VOCdevkit/VOC2012/JPEGImages/',
                            modifier=lambda x: x + '.jpg',
                            use_bgr=True,
                            mean=np.array((128., 128., 128.)))
sess.run(sir.initializers)

while True:
    try:
        path, image = sess.run([sir.path, sir.image])
        print(path, image.shape)
    except tf.errors.OutOfRangeError as e:
        print(e.message)
        break
