# @Time    : 2018/7/26 23:05
# @Author  : cap
# @FileName: xl_eval.py
# @Software: PyCharm Community Edition
# @introduction:
import time
from datetime import datetime
import tensorflow as tf
import numpy as np

import xl_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'D:\\softfiles\\workspace\\games\\xue_lang\\eval',
                           """where to write event logs""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'D:\\softfiles\\workspace\\games\\xue_lang\\models',
                           """checkpoint dir""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, """how often to run the eval""")
tf.app.flags.DEFINE_integer('num_examples', 400, """number of example""")
tf.app.flags.DEFINE_boolean('run_once', False, """run once ?""")

def eval_once(saver, summary_writer, top_k_op, summary_op, iterator):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        sess.run(iterator.initializer)
        # FLAGS.num_examples / 50
        step = 0
        true_count = 0
        for _ in range(14):
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
        precision = true_count / 400
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision @ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)

def evaluate():
    with tf.Graph().as_default() as g:
        images, labels, iterator = xl_model.inputs()

        logits = xl_model.inference(images, False)
        logits = tf.nn.softmax(logits)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_averages = tf.train.ExponentialMovingAverage(xl_model.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        # with tf.Session() as sess:
        #     sess.run(iterator.initializer)
        #     ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        #     if ckpt and ckpt.model_checkpoint_path:
        #         saver.restore(sess, ckpt.model_checkpoint_path)
        #         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        #     print(sess.run([logits, labels, top_k_op]))
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op, iterator)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(_):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
