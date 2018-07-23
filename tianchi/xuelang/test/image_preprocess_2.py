# @Time    : 2018/7/21 17:07
# @Author  : cap
# @FileName: image_preprocess_2.py
# @Software: PyCharm Community Edition
# @introduction:
import os
import pickle

import numpy as np
import tensorflow as tf

import find_jpg


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


label_map_dict_dir = './label_map.pickle'

# 获取文件列表
test_list_files, train_dict_file, xmls = find_jpg.get_file_names('D:\\softfiles\\workspace\\games\\xuelang')

# 把数据保存成tfrecord格式
# tfrecord{image_raw, label, pixels}
with open(label_map_dict_dir, 'rb') as f:
    label_map_dict = pickle.load(f)

train_x = None # example{}
train_y = None
test_x = None
test_y = None
pred_x = None

# 确定训练样本的数量
file_nums = 0
for key, value in train_dict_file.items():
    file_nums += len(value)
print('训练样本的数量为：%s' % file_nums)

# train.tfrecord
record_name = './train.tfrecord'
writer = tf.python_io.TFRecordWriter(record_name)
count = 0

sess = tf.Session()
## 按照标注框截取指定训练区域
# 确定标注框的位置（offset_height, offset_width, target_height, target_width）
for label, pathnames in train_dict_file.items():
    for pathname in pathnames:
        count += 1
        print(label, count, pathname)
        # 选出绝对路径的最后的文件名，并去除扩展名.jpg
        filename = pathname.split(os.sep)[-1][:-4]
        # 读取图片文件，并转成图片格式
        image_raw_data = tf.gfile.FastGFile(pathname, 'rb').read()
        img_data = tf.image.decode_jpeg(image_raw_data)
        #
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
        # 获取标注框信息,如果读取到的信息为None，则使用原图信息
        xml = xmls.get(filename, None)
        if xml:
            offset_height = int(xml['ymin'])
            offset_width = int(xml['xmin'])
            target_height = int(xml['ymax']) - offset_height
            target_width = int(xml['xmax']) - offset_width
            img_data = tf.image.crop_to_bounding_box(img_data, offset_height, offset_width, target_height, target_width)

        # pixels

        shape = sess.run(img_data).shape
        #
        img_data = tf.image.convert_image_dtype(img_data, tf.uint8)
        image_raw = sess.run(img_data).tostring()
        # print(np.array(im).tostring())6
        # image_raw = tf.image.encode_jpeg(img_data)

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(shape[0]),
            'width': _int64_feature(shape[1]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)
        }))

        writer.write(example.SerializeToString())
writer.close()
sess.close()