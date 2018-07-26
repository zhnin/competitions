# @Time    : 2018/7/25 19:36
# @Author  : cap
# @FileName: xl_train.py
# @Software: PyCharm Community Edition
# @introduction:
import os
from datetime import datetime
import time

import tensorflow as tf

import xl_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'D:\\softfiles\\workspace\\games\\xue_lang\\models',
                           """directory where write event log and checkpoint!""")
tf.app.flags.DEFINE_integer('max_steps', 10000, """max steps""")
tf.app.flags.DEFINE_integer('log_frequency', 10, """log result to console.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")


# 滑动平均下降， 学习率， 学习率下降
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_DECAY_FACTOR = 0.99
INITIAL_LEARNING_RATE = 0.1


def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # 获取训练数据迭代器
        with tf.device('/cpu:0'):
            images, labels, iterator = xl_model.distorted_inputs()

        logits = xl_model.inference(images)

        loss = xl_model.loss(logits, labels)

        train_op = xl_model.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        ) as sess:
            # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            sess.run(iterator.initializer)
            while not sess.should_stop():
                sess.run(train_op)


        # with tf.Session() as sess:
        #     tf.global_variables_initializer().run()
        #         # paths = tf.train.match_filenames_once('D:\\softfiles\\workspace\\games\\xue_lang\\prep_data\\60_60\\train_*.record')
        #     # train_paths = [os.path.join(FLAGS.data_dir, 'train_%d_4.record') % i for i in range(1, 5)]
        #     # print(train_paths)
        #     sess.run(iterator.initializer)
        #     # sess.run(test_iterator.initializer, feed_dict={testfile: [test_path]})
        #     # print(test_images.get_shape())
        #
        #     # a = sess.run(images)[0,...]
        #     # import matplotlib.pyplot as mp
        #     # mp.imshow(a)
        #     # mp.show()
        #
        #     for i in range(1000):
        #         _, loss_value, step = sess.run([train_op, loss, global_step])
        #         if i % 100 == 0:
        #             print('After %d steps：losses: %g'% (step, loss_value))

def main(_):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
