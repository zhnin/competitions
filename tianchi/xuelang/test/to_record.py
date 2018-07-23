# @Time    : 2018/7/23 10:18
# @Author  : cap
# @FileName: to_record.py
# @Software: PyCharm Community Edition
# @introduction:
# 把原始数据保存成tfrecord格式，分别为，train_x, train_y, test_x, test_y, pred_x
"""
1. 获取原始数据列表
2. 获取每个类别的样本数量，每个label要有1000个左右的样本，不够的样本，依据用随机标注框选取0.3标注区域
    如果一个类别有30个样本，则每个样本需要额外扩1000//30，即每个样本额外扩33-1张图，最后把该类别按照随机二八比例依次写入
    train和test中
    train_x, train_y --> train.tfrecord
    test_x, test_y --> test.tfrecord
    pred_x -- >pred.tfrecord
3. 最终得到三个tfrecord文件
"""
import os
import random
import numpy as np
import tensorflow as tf

import find_jpg

min_sample_per_label = 500

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_filenames(path):
    test_list_files, train_dict_file, xmls = find_jpg.get_file_names(path)
    return test_list_files, train_dict_file, xmls


def distort_color(image):
    brightness = lambda image: tf.image.random_brightness(image, max_delta= 32. / 255.)
    saturation = lambda image: tf.image.random_saturation(image, lower=0.5, upper=1.5)
    hue = lambda image: tf.image.random_hue(image, max_delta=0.2)
    contrast = lambda image: tf.image.random_contrast(image, lower=0.5, upper=1.5)

    func = [brightness, saturation, hue, contrast]
    rand_index = random.sample([0,1,2,3],4)

    for i in rand_index:
        image = func[rand_index[i]](image)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=(1, 1, 4))

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, tf.float32)

    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))

    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_flip_up_down(distorted_image)

    distorted_image = distort_color(distorted_image)
    return distorted_image


def train_process(train_dict_file, xmls, sess, height, width):
    # 定义两个tfrecord writer
    train_writer = tf.python_io.TFRecordWriter('./train.tfrecord')
    test_writer = tf.python_io.TFRecordWriter('./test.tfrecord')
    # 类别的数量
    len_label = len(train_dict_file)
    label_count = 0
    for label, files in train_dict_file.items():
        file_count = 0
        label_count += 1
        # 获取类别样本数量,如果小于每个类别的最小样本数量限制，则开始样本扩充操作
        num_sample = len(files)
        loops = 1
        example_list = []
        if num_sample < min_sample_per_label:
            loops = min_sample_per_label // num_sample

        # 读取样本，
        # decode convert，decode时图像做八倍缩小处理
        for file in files:
            filename = file.split(os.sep)[-1][:-4]

            # 构建标注框信息
            xml = xmls.get(filename, None)
            # 如果取到标注框，创建box，如果娶不到，则用原图（缩放过）
            boxes = None
            if xml:
                box = [float(xml['ymin']) / float(xml['height']), float(xml['xmin']) / float(xml['width']), \
                       int(xml['ymax']) / float(xml['height']), int(xml['xmax']) / float(xml['width'])]
                boxes = tf.constant(box, shape=(1, 1, 4), dtype=tf.float32)

                # offset_height = int(xml['ymin'])
                # offset_width = int(xml['xmin'])
                # target_height = int(xml['ymax']) - offset_height
                # target_width = int(xml['xmax']) - offset_width
                # image_raw = tf.image.crop_to_bounding_box(image_raw, offset_height, offset_width, target_height,
                #                                          target_width)
            for _ in range(loops):
                image_raw = tf.gfile.FastGFile(file, 'rb').read()
                image_raw = tf.image.decode_jpeg(image_raw, channels=3, ratio=8, dct_method='INTEGER_FAST')
                image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)
                image_raw = preprocess(image_raw, height, width, boxes)
                image_raw = tf.image.convert_image_dtype(image_raw, dtype=tf.uint8)
                image_raw = tf.image.encode_jpeg(image_raw, format='rgb', quality=50)
                sample = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(label),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'image_raw': _bytes_feature(sess.run(image_raw))
                }))
                example_list.append(sample)
                file_count += 1
                print('\r类别%d/%d,正在处理文件:%d' % (label_count, len_label, file_count), end='')
        # 对example按照二八比例随机划分
        print('\r开始写入:', label, end='')
        len_example = len(example_list)
        random_list = random.sample(list(range(len_example)), len_example)
        train_index = random_list[:int((len_example * 0.8))]
        test_index = random_list[int((len_example * 0.8)):]
        for i in train_index:
            train_writer.write(example_list[i].SerializeToString())
        for j in test_index:
            test_writer.write(example_list[j].SerializeToString())
        print('\r写入完成：', label, end='')
    train_writer.close()
    test_writer.close()


def pred_process(test_list_files, sess, height, width):
    writer = tf.python_io.TFRecordWriter('./pred.tfrecord')
    count = 1
    num = len(test_list_files)
    for file in test_list_files:
        print('\r%d/%d' % (count, num), end='')
        image_raw = tf.gfile.GFile(file, 'rb').read()
        image_raw = tf.image.decode_jpeg(image_raw, channels=3, ratio=8, dct_method='INTEGER_FAST')
        image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)
        distorted_image = tf.image.resize_images(image_raw, [height, width], method=np.random.randint(4))
        distorted_image = tf.image.convert_image_dtype(distorted_image, dtype=tf.uint8)
        distorted_image = tf.image.encode_jpeg(distorted_image, format='rgb', quality=50)
        sample = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(sess.run(distorted_image))
        }))
        writer.write(sample.SerializeToString())
        count += 1
    writer.close()


def main():
    dir_path = 'D:\\softfiles\\workspace\\games\\xuelang'
    test_list_files, train_dict_file, xmls = get_filenames(dir_path)

    height = 240
    width = 320
    with tf.Session() as sess:
        train_process(train_dict_file, xmls, sess, height, width)
        # pred_process(test_list_files, sess, height, width)



if __name__ == '__main__':
    main()
