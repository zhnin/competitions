# @Time    : 2018/7/25 19:36
# @Author  : cap
# @FileName: xl_train.py
# @Software: PyCharm Community Edition
# @introduction:
import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'D:\\softfiles\\workspace\\games\\prep_data', 'data dir')
tf.app.flags.DEFINE_string('model_dir', 'D:\\softfiles\\workspace\\games\\models\\xuelang', 'model dir')
tf.app.flags.DEFINE_integer('batch_size', 10, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 100, 'epoch')

TRAIN_NUMBER = 1617
TEST_NUMBER = 405
PRED_NUMBER = 662

IMAGE_SIZE = 200
BUFFER_SIZE = 100

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

    # image_raw = tf.random_crop(image_raw, [IMAGE_SIZE, IMAGE_SIZE, 3])
    # image_raw = tf.image.random_flip_left_right(image_raw)

    # image_raw = tf.image.random_brightness(image_raw, max_delta=63)
    # image_raw = tf.image.random_contrast(image_raw, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    # image_raw = tf.image.per_image_standardization(image_raw)

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
    image_raw = tf.random_crop(image_raw, [IMAGE_SIZE, IMAGE_SIZE, 3])
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
        dataset = dataset.map(_parse_train_function).batch(FLAGS.batch_size).repeat(FLAGS.epoch)
    else:
        dataset = dataset.map(_parse_test_function).batch(FLAGS.batch_size)

    iterator = dataset.make_initializable_iterator()
    return iterator


def inference(images):
    """
    conv1: 200*200*3
    :param images:
    :return:
    """
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
        # kernel = tf.get_variable('weights', shape=[5, 5, 3, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        # biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('local3', reuse=tf.AUTO_REUSE) as scope:
        pool2_shape = pool2.get_shape().as_list()
        new_dim = int((IMAGE_SIZE / 4) * (IMAGE_SIZE / 4) * pool2_shape[3])
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        # print(dim)
        weights = _variable_with_weight_decay('weights', shape=[new_dim, 384], stddev=0.04, wd=0.04)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    with tf.variable_scope('local4', reuse=tf.AUTO_REUSE) as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    with tf.variable_scope('softmax_linear', reuse=tf.AUTO_REUSE) as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        # 获取训练数据迭代器
        with tf.device('/cpu:0'):
            filenames = tf.placeholder(tf.string, shape=[None])
            iterator = input_reader(filenames, eval=False)
            images, labels = iterator.get_next()

            # testfile = tf.placeholder(tf.string, shape=[None])
            # test_iterator = input_reader(testfile, eval=True)
            # test_images, test_labels = test_iterator.get_next()

        logits = inference(images)

        losses = loss(logits, labels)

        # num_batches_per_epoch = PRED_NUMBER / FLAGS.batch_size
        # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        1000,
                                        LEARNING_RATE_DECAY_FACTOR)

        tf.summary.scalar('learing_rate', lr)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        grad_op = tf.train.GradientDescentOptimizer(lr).minimize(losses, global_step)

        with tf.control_dependencies([variable_averages_op, grad_op]):
            train_op = tf.no_op(name='train')

        # 定义测试数据集
        #

        # test_logits = inference(test_images)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.float32))



        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            test_path = os.path.join(FLAGS.data_dir, 'test.record')
            train_path = os.path.join(FLAGS.data_dir, 'train.record')

            sess.run(iterator.initializer, feed_dict={filenames: [train_path]})
            # sess.run(test_iterator.initializer, feed_dict={testfile: [test_path]})
            # print(test_images.get_shape())
            print(sess.run(images)[0,...])
            # for i in range(10000):
            #     sess.run(train_op)
            #     if i % 10 == 0:
            #         lo, ac = sess.run([losses, accuracy])
            #         print('After %d steps：losses: %.2f, accuracy: %.2f' % (i, lo, ac))


                # if i % 100 == 0:
                #     accu_list = []
                #     sess.run(test_iterator.initializer, feed_dict={testfile: [test_path]})
                #     for j in range(40):
                #         accu_list.append(sess.run(accuracy))
                #     accu = sum(accu_list) / len(accu_list)
                #     print('After %d steps, accu: %.3f' % (j, accu))




def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
