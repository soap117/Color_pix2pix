import os
import time
from PIL import Image
import cv2 as cv
import numpy as np
import tensorflow as tf

OUTPUT_SIZE = 128
DEPTH = 3


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_path, name):
    """
    Converts s dataset to tfrecords
    """
    rows = OUTPUT_SIZE*2
    cols = OUTPUT_SIZE
    depth = DEPTH
    for ii in range(10):
        writer = tf.python_io.TFRecordWriter(name + str(ii) + '.tfrecords')
        len = os.listdir(data_path).__len__()
        size_each = int(len/10)
        for img_name in os.listdir(data_path)[ii * size_each: (ii + 1) * size_each]:
            img_path = data_path + img_name
            img = Image.open(img_path)
            img = np.asarray(img)
            img = cv.resize(img,(256,128))
            img = Image.fromarray(np.uint8(img))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'image_raw': _bytes_feature(img_raw)}))
            writer.write(example.SerializeToString())
        writer.close()


if __name__ == '__main__':
    current_dir = os.getcwd()
    data_path = input("Enter the original data location: ");
    name = input("Where you want to save the new data: ");
    start_time = time.time()

    print('Convert start')
    print('\n' * 2)

    convert_to(data_path, name)

    print('\n' * 2)
    print('Convert done, take %.2f seconds' % (time.time() - start_time))
