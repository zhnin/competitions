# @Time    : 2018/7/25 14:26
# @Author  : cap
# @FileName: image2tfrecord2.py
# @Software: PyCharm Community Edition
# @introduction:
import os
import random
import xml.etree.ElementTree as ET

import tensorflow as tf
import sklearn.model_selection as sm


ratio = 8
height = 240
width = 320

def find_pred_path(pred_path):
    pred_list = []
    for pred_dir in pred_path:
        for cur_dir, sub_dir, files in os.walk(pred_dir):
            if sub_dir == [] and len(files) > 0:
                for file in files:
                    if file.endswith('.jpg'):
                        pred_list.append(os.path.join(cur_dir, file))
    return pred_list


def find_train_path(train_dir_path):
    # 分别用于存放正常文件路径，和瑕疵问题图片路径
    train_list = []
    label_list = []
    for train_dir in train_dir_path:
        for cur_dir, sub_dir, files in os.walk(train_dir):
            if sub_dir == [] and len(files) > 0:
                for file in files:
                    if file.endswith('.jpg'):
                        train_list.append(os.path.join(cur_dir, file))
                        if cur_dir.split(os.sep)[-1] == '正常':
                            label_list.append(0)
                        else:
                            label_list.append(1)
    return train_list, label_list


def getbox(filepath):
    xml = {}
    filepath = filepath.replace('.jpg', '.xml')
    root = ET.parse(filepath).getroot()
    for ele in root.iter():
        xml[ele.tag] = ele.text
    return xml


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_record(x, y, record_name):
    count = 0
    with tf.device('/cpu:0'):
        writer = tf.python_io.TFRecordWriter(record_name)
        sess = tf.Session()
        for filepath, label in zip(x, y):
            image_raw = tf.gfile.GFile(filepath, 'rb').read()
            image_raw = tf.image.decode_jpeg(image_raw, channels=3)
            image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)

            # 获取标注框
            if filepath.split(os.sep)[-2] != '正常' and label is not None:
                xml = getbox(filepath)
                offset_height = int(xml['ymin'])
                offset_width = int(xml['xmin'])
                target_height = int(xml['ymax']) - offset_height
                target_width = int(xml['xmax']) - offset_width

                image_raw = tf.image.crop_to_bounding_box(image_raw,
                                                           offset_height, offset_width,
                                                           target_height,
                                                           target_width)

            image_raw = tf.image.resize_image_with_crop_or_pad(image_raw, 240, 320)
            image_raw = tf.image.convert_image_dtype(image_raw, tf.uint8)

            image_raw = tf.image.encode_jpeg(image_raw, format='rgb', quality=98)
            image_raw = sess.run(image_raw)
            if label in [0, 1]:
                sample = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'label': _int64_feature(label),
                    'image_raw': _bytes_feature(image_raw)
                }))
            else:
                sample = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'image_raw': _bytes_feature(image_raw)
                }))
            writer.write(sample.SerializeToString())
            del sample
            del image_raw
            count += 1
            print('\rSuccessful write %d.' % count, end='')
        writer.close()
        print('\nSuccessfully!!')
        sess.close()
        print('close')


def main():
    # dest_dir = 'D:\\softfiles\\workspace\\games\\xuelang'
    # train_dirs = ['xuelang_round1_train_part1_20180628',
    #               'xuelang_round1_train_part2_20180705',
    #               'xuelang_round1_train_part3_20180709']
    # pred_dirs = ['xuelang_round1_test_a_20180709']
    #
    # train_dir_path = [os.path.join(dest_dir, i) for i in train_dirs]
    # pred_dir_path = [os.path.join(dest_dir, i) for i in pred_dirs]
    #
    # train_list, label_list = find_train_path(train_dir_path)
    # pred_list = find_pred_path(pred_dir_path)
    # train_x, test_x, train_y, test_y = sm.train_test_split(train_list, label_list, test_size=0.2, random_state=10)
    import pickle
    with open('train_x', 'rb') as f:
        train_x = pickle.load(f)
    with open('train_y', 'rb') as f:
        train_y = pickle.load(f)
    with open('test_x', 'rb') as f:
        test_x = pickle.load(f)
    with open('test_y', 'rb') as f:
        test_y = pickle.load(f)



    # to_record(train_x[:400], train_y[:400], 'train_1_4.record')
    # to_record(train_x[400:800], train_y[400:800], 'train_2_4.record')
    # to_record(train_x[800:1200], train_y[800:1200], 'train_3_4.record')
    # to_record(train_x[1200:1600], train_y[1200:1600], 'train_4_4.record')
    to_record(test_x, test_y, 'test.record')
    print(len(train_x))
    print(len(train_y))
    print(len(test_x))
    print(len(test_y))
    # print(len(pred_list))
    # to_record(train_x, train_y, 'train.record')
    # to_record(test_x, test_y, 'test.record')
    # to_record(pred_list, [None for _ in range(len(pred_list))], 'pred.record')

if __name__ == '__main__':
    main()
