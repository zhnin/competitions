# @Time    : 2018/7/26 19:35
# @Author  : cap
# @FileName: xl_input.py
# @Software: PyCharm Community Edition
# @introduction:
import os

import tensorflow as tf

# 训练数据集， 测试数据集， 预测数据集
TRAIN_NUMBER = 1600
TEST_NUMBER = 400
PRED_NUMBER = 662

# 训练数据集和测试数据集图片大小
IMAGE_SIZE = 60

# 输出类别，1：有问题。0:正常
NUM_CLASSES = 2

def _parse_train_function(record):
    parse_feature = tf.parse_single_example(record, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })

    label = parse_feature['label']
    image_raw = parse_feature['image_raw']

    # 解析成图片，并将格式转成tf.float32
    image_raw = tf.image.decode_jpeg(image_raw, channels=3)
    image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)

    # 左右转换
    # image_raw = tf.image.random_flip_left_right(image_raw)

    # 亮度，对比度随机调整
    # image_raw = tf.image.random_brightness(image_raw, max_delta=63)
    # image_raw = tf.image.random_contrast(image_raw, lower=0.2, upper=1.8)

    # 数值均化处理
    # image_raw = tf.image.per_image_standardization(image_raw)
    # 输出值调整
    # image_raw = tf.clip_by_value(image_raw, 0.0, 1.0)
    image_raw.set_shape([240, 320, 3])
    return image_raw, label


def _parse_eval_function(record):
    parse_feature = tf.parse_single_example(record, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })

    label = parse_feature['label']
    image_raw = parse_feature['image_raw']

    # 解析成图片，并将格式转成tf.float32
    image_raw = tf.image.decode_jpeg(image_raw, channels=3)
    image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)

    image_raw.set_shape([240, 320, 3])
    return image_raw, label


def distorted_inputs(data_dir, batch_size, epoch):
    filenames = tf.train.match_filenames_once(data_dir)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_train_function).batch(batch_size).repeat(epoch)

    #  创建迭代器
    iterator = dataset.make_initializable_iterator()
    return iterator


def inputs(data_dir, batch_size):
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(data_dir)
        dataset = dataset.map(_parse_eval_function).batch(batch_size)
        iterator = dataset.make_initializable_iterator()
    return iterator