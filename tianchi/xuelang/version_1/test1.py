# @Time    : 2018/7/28 1:21
# @Author  : cap
# @FileName: test1.py
# @Software: PyCharm Community Edition
# @introduction:
import os
import tensorflow as tf
import xml.etree.ElementTree as ET
import matplotlib.pyplot as mp
import random
import xl_model

def getbox(filepath):
    xml = {}
    filepath = filepath.replace('.jpg', '.xml')
    root = ET.parse(filepath).getroot()
    for ele in root.iter():
        xml[ele.tag] = ele.text
    return xml

def box(filepath):
    ratio = 1
    xml = getbox(filepath)
    offset_height = int(int(xml['ymin']) / ratio)
    offset_width = int(int(xml['xmin']) / ratio)
    target_height = int(int(xml['ymax']) / ratio) - offset_height
    target_width = int(int(xml['xmax']) / ratio) - offset_width
    return offset_height, offset_width,target_height, target_width

def test():
    # filenames = 'D:\\softfiles\\workspace\\games\\xuelang\\xuelang_round1_train_part1_20180628\\扎洞'
    filenames = 'D:\\softfiles\\workspace\\games\\xuelang\\xuelang_round1_train_part1_20180628\\正常'
    file_list = []
    for _, _, files in os.walk(filenames):
        for file in files:
            if file.endswith('.jpg'):
                file_list.append(os.path.join(filenames, file))

    # print(file_list)

    file = file_list[random.randint(0, 19)]
    image_raw = tf.gfile.GFile(file, 'rb').read()
    image_raw = tf.image.decode_jpeg(image_raw)

    image_raw = tf.image.convert_image_dtype(image_raw, tf.float32)

    # boxes = box(file)
    # print(boxes)

    # image_raw = tf.image.crop_to_bounding_box(image_raw, *boxes)
    image_raw = tf.image.resize_image_with_crop_or_pad(image_raw, 240, 320)
    # print(image_raw)

    with tf.Session() as sess:
        print(image_raw.get_shape())
        image = sess.run(image_raw)
        # print(sess.run(image_raw))
        mp.imshow(image)
        mp.title(file.split(os.sep)[-1])
        mp.show()

def test2():
    filenames = 'D:\\softfiles\\workspace\\games\\xuelang\\xuelang_round1_train_part1_20180628\\扎洞'
    file_list = []
    for _, _, files in os.walk(filenames):
        for file in files:
            if file.endswith('.jpg'):
                file_list.append(os.path.join(filenames, file))

    # print(file_list)

    file = file_list[8]
    dataset = tf.data.TextLineDataset(file)
    # dataset = dataset.make_one_shot_iterator()
    # image_raw = dataset.get_next()
    with tf.Session() as sess:
        print(dataset.get_shape())
        print(sess.run(dataset))

def test3():
    pass

test()