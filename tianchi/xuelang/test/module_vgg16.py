# @Time    : 2018/7/23 19:59
# @Author  : cap
# @FileName: module_vgg16.py
# @Software: PyCharm Community Edition
# @introduction:

import tensorflow as tf
import numpy as np


def parse(record):
    features = tf.parse_single_example(record, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    image_raw = tf.image.decode_jpeg(features['image_raw'])

def main(_):
    """
    创建文件列表

    :param _:
    :return:
    """
    # files = tf.train.match_filenames_once('./*.tfrecord')
    files = ['./train']
    print(files)


if __name__ == '__main__':
    tf.app.run()
