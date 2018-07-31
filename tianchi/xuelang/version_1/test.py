# @Time    : 2018/7/27 14:23
# @Author  : cap
# @FileName: test.py
# @Software: PyCharm Community Edition
# @introduction:
import tensorflow as tf
import os
import matplotlib.pyplot as mp

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
    # image_raw.set_shape([60, 60, 3])
    # image_raw = tf.image.rgb_to_grayscale(image_raw)
    # image_raw = tf.image.
    # tf.image.rgb
    return image_raw, label

data_dir = os.path.join('D:\\softfiles\\workspace\\games\\xue_lang\\prep_data\\240_320', 'train_*.record')
filenames = tf.train.match_filenames_once(data_dir)
dataset = tf.data.TFRecordDataset(filenames)

dataset = dataset.map(_parse_train_function).shuffle(100).batch(1)

iterator = dataset.make_initializable_iterator()
images, labels = iterator.get_next()
images = tf.image.convert_image_dtype(images, tf.uint8)

with tf.Session() as sess:
    sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
    sess.run(iterator.initializer)

    while True:
        image, label = sess.run([images, labels])
        if label == 1:
            break
    import cv2 as cv
    img = image[0,...]
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('gray', gray)
    # print(gray.shape)
    # cv.waitKey()
    # mp.imshow(img[...,0])
    #
    mp.imshow(img)
    mp.title(label)
    mp.show()
