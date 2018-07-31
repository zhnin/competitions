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


def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # 获取训练数据迭代器
        with tf.device('/cpu:0'):
            images, labels, iterator = xl_model.distorted_inputs()
        logits = xl_model.inference(images, False)
        softmax_logits = tf.nn.softmax(logits)
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
            sess.run([iterator.initializer])
            # while not sess.should_stop():
            #     sess.run(train_op)

            # for i in range(5):
            #     # _, los = sess.run([train_op, loss])
            #     _, logit, label = sess.run([train_op, softmax_logits, labels])
            #     # print(logits, label)
            #     for lo, la in zip(logit, label):
            #         print(lo, la)


            # import matplotlib.pyplot as mp
            #
            # _, im, l, si = sess.run([train_op,images, labels, si_im])
            # print(si)
            # mp.imshow(si[0,...])
            # mp.title(l[0])
            # mp.show()

def main(_):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
