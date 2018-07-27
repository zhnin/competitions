# @Time    : 2018/7/26 19:35
# @Author  : cap
# @FileName: xl_model.py
# @Software: PyCharm Community Edition
# @introduction:

import os
import sys

import tensorflow as tf

import xl_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'D:\\softfiles\\workspace\\games\\xue_lang\\prep_data\\240_320', 'data dir')
tf.app.flags.DEFINE_integer('batch_size', 25, 'batch size')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'uding fp16')
tf.app.flags.DEFINE_integer('epoch', 100, 'epoch')

IMAGE_SIZE = xl_input.IMAGE_SIZE
NUM_CLASSES = xl_input.NUM_CLASSES
TRAIN_NUMBER = xl_input.TRAIN_NUMBER
TEST_NUMBER = xl_input.TEST_NUMBER

# 滑动平均下降， 学习率， 学习率下降
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_DECAY_FACTOR = 0.99
INITIAL_LEARNING_RATE = 0.1


def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('data dir can not found.')
    data_dir = os.path.join(FLAGS.data_dir, 'train_*.record')
    iterator = xl_input.distorted_inputs(data_dir, FLAGS.batch_size, FLAGS.epoch)
    images, labels = iterator.get_next()
    #
    if FLAGS.use_fp16:
        images = tf.image.convert_image_dtype(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels, iterator


def inputs():
    if not FLAGS.data_dir:
        raise ValueError('data dir is not exist.')
    data_dir = os.path.join(FLAGS.data_dir, 'test.record')
    iterator = xl_input.inputs(data_dir, FLAGS.batch_size)
    images, labels = iterator.get_next()
    if FLAGS.use_fp16:
        images = tf.image.convert_image_dtype(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels, iterator


def _activation_summary(x):
    name = x.op.name
    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images, train):
    """
    conv1: 60*60*3      60*60*64
    pool1: 60*60*64     30*30*64
    norm1:

    conv2: 30*30*64     30*30*256
    norm2:
    pool2: 30*30*256    15*15*256

    conv3: 15*15*256    15*15*512
    norm3:
    pool3: 15*15*512   8*8*512

    local4:

    local5
    """
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 16], stddev=0.01, wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 16, 32], stddev=0.01, wd=None)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 32, 64], stddev=0.01, wd=None)
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3)

    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('local4') as scope:
        shape = pool3.get_shape().as_list()
        shape_size = shape[1] * shape[2] * shape[3]
        reshape = tf.reshape(pool3, [FLAGS.batch_size, shape_size])

        weights = _variable_with_weight_decay('weights', shape=[shape_size, 256], stddev=0.04, wd=0.04)
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.01))
        local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        if train:
            local4 = tf.nn.dropout(local4, 0.5)
        _activation_summary(local4)

    with tf.variable_scope('local5') as scope:
        weights = _variable_with_weight_decay('weights', shape=[256, 64], stddev=0.04, wd=0.04)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.01))
        local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)
        if train:
            local5 = tf.nn.dropout(local5, 0.5)
        _activation_summary(local5)

    with tf.variable_scope('local6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[64, NUM_CLASSES], stddev=0.04, wd=0.04)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.01))
        local6 = tf.matmul(local5, weights) + biases
        _activation_summary(local6)

    return local6


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(losses, global_step):
    # 学习率优化
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    50,
                                    LEARNING_RATE_DECAY_FACTOR)

    tf.summary.scalar('learing_rate', lr)

    # 滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    grad_op = tf.train.GradientDescentOptimizer(lr).minimize(losses, global_step)

    with tf.control_dependencies([grad_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

