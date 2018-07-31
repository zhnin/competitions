# @Time    : 2018/7/28 19:55
# @Author  : cap
# @FileName: test2.py
# @Software: PyCharm Community Edition
# @introduction:
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as mp

path = 'D:\\softfiles\\workspace\\games\\xuelang\\xuelang_round1_train_part3_20180709\\吊经\\J01_2018.06.19 13_37_56.jpg'

image_raw = tf.gfile.GFile(path, 'rb').read()

image_raw = tf.image.decode_jpeg(image_raw)
image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)

# image_raw = tf.image.rgb_to_grayscale(image_raw)

image_raw = tf.image.convert_image_dtype(image_raw, tf.uint8)
with tf.Session() as sess:
    image = sess.run(image_raw)

    mp.imshow(image[...,0])

    mp.show()




