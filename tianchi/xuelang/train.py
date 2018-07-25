# @Time    : 2018/7/25 19:36
# @Author  : cap
# @FileName: train.py
# @Software: PyCharm Community Edition
# @introduction:
import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'D:\\softfiles\\workspace\\games\\prep_data', 'data dir')
tf.app.flags.DEFINE_string('model_dir', 'D:\\softfiles\\workspace\\games\\models\\xuelang', 'model dir')
tf.app.flags.DEFINE_integer('batch_size', 100, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 50, 'epoch')

TRAIN_NUMBER = 1617
TEST_NUMBER = 405
PRED_NUMBER = 662

IMAGE_SIZE = 200
BUFFER_SIZE = 1000

MOVING_AVERAGE_DECAY = 0.99
NUM_EPOCHS_PER_DECAY = 100
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

NUM_CLASSES = 2

def _activation_summary(x):
    name = x.op.name
    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))


def _parse_train_function(record):
    parse_feature = tf.parse_single_example(record, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })

    label = parse_feature['label']
    image_raw = parse_feature['image_raw']

    image_raw = tf.image.decode_jpeg(image_raw, channels=3)
    image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)

    image_raw = tf.random_crop(image_raw, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image_raw = tf.image.random_flip_left_right(image_raw)

    image_raw = tf.image.random_brightness(image_raw, max_delta=63)
    image_raw = tf.image.random_contrast(image_raw, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    image_raw = tf.image.per_image_standardization(image_raw)

    # Set the shapes of tensors.
    image_raw.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    return image_raw, label


def _parse_test_function(record):
    parse_feature = tf.parse_single_example(record, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })

    label = parse_feature['label']
    image_raw = parse_feature['image_raw']

    image_raw = tf.image.decode_jpeg(image_raw, channels=3)
    image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)
    return image_raw, label


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


def input_reader(filenames, eval=False):
    dataset = tf.data.TFRecordDataset(filenames)
    if not eval:
        dataset = dataset.map(_parse_train_function).shuffle(BUFFER_SIZE).batch(FLAGS.batch_size).repeat(FLAGS.epoch)
    else:
        dataset = dataset.map(_parse_test_function).batch(TEST_NUMBER)

    iterator = dataset.make_initializable_iterator()
    return iterator


def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, bias=biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weight', shape=[dim, 384], stddev=0.04, wd=0.04)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):


def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
    # 获取训练数据迭代器
    with tf.device('/cpu:0'):
        filenames = tf.placeholder(tf.string, shape=[None])
        iterator = input_reader(filenames, eval=False)
        images, labels = iterator.get_next()

    logits = inference(images)

    with tf.Session() as sess:
        path = os.path.join(FLAGS.data_dir, 'test.record')
        sess.run(iterator.initializer, feed_dict={filenames: [path]})
        print(sess.run([labels]))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
