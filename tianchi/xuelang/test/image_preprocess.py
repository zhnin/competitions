# @Time    : 2018/7/21 15:20
# @Author  : cap
# @FileName: image_preprocess.py
# @Software: PyCharm Community Edition
# @introduction: 对train数据按照标注框进行裁剪，并进行灰度处理

# 方法1，采用tf.image.sample_distorted_bounding_box
import tensorflow as tf
import matplotlib.pyplot as mp
import os
import pickle
import find_jpg

test_list_files, train_dict_file, xmls = find_jpg.get_file_names('D:\\softfiles\\workspace\\games\\xuelang')

def label_map(train_dict_file):

    # 设置编码格式，并保存编码字典
    ab = []
    for key, value in train_dict_file.items():
        ab.append((key, len(value)))
    sorted_ab = sorted(ab, key=lambda x:x[1], reverse=True)

    # 保存label映射字典
    label_map = {}
    for key, value in zip(range(len(sorted_ab)), sorted_ab):
        label_map[key] = value[0]
    with open('./label_map.pickle', 'wb') as f:
        pickle.dump(label_map, f)

    # 对trainlabel进行映射


# D:\softfiles\workspace\games\xuelang\xuelang_round1_train_part2_20180705\跳花\J01_2018.06.13 13_19_45.jpg
file_path = 'D:\\softfiles\\workspace\\games\\xuelang\\xuelang_round1_train_part2_20180705\\跳花\\J01_2018.06.22 10_38_20.jpg'
# print(xmls['J01_2018.06.22 10_38_20'])
# {'annotation': '\n\t', 'filename': 'J01_2018.06.13 13_19_45.jpg',
# 'source': '\n\t\t', 'database': 'Unknown', 'size': '\n\t\t',
# 'width': '2560', 'height': '1920', 'depth': '3', '
# segmented': '0', 'object': '\n\t\t', 'name': '跳花', 'pose': 'Unspecified',
# 'truncated': '0', 'difficult': '0', 'bndbox': '\n\t\t\t',
# 'xmin': '1281', 'ymin': '537', 'xmax': '1344', 'ymax': '820'}
xml = {'annotation': '\n\t', 'filename': 'J01_2018.06.22 10_38_20.jpg',
       'source': '\n\t\t', 'database': 'Unknown', 'size': '\n\t\t',
       'width': '2560', 'height': '1920', 'depth': '3', 'segmented': '0', 'object': '\n\t\t',
       'name': '跳花', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': '\n\t\t\t',
       'xmin': '591', 'ymin': '90', 'xmax': '785', 'ymax': '1026'}

image_raw_data = tf.gfile.FastGFile(file_path, 'rb').read()



with tf.Session() as sess:
    #
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # img_data = tf.expand_dims(img_data, 0)
    # 增加标注框
    box = [float(xml['ymin']) / float(xml['height']), float(xml['xmin']) / float(xml['width']), \
          int(xml['ymax']) / float(xml['height']), int(xml['xmax']) / float(xml['width'])]
    boxes = tf.constant(box, shape=(1, 4), dtype=tf.float32)

    # boxes [batch, height, width, channel]
    boxes = tf.expand_dims(boxes, 0)
    # print(boxes.eval().shape)
    # result = tf.image.draw_bounding_boxes(img_data, boxes)
    # # print('shape: ', img_data.eval().shape)
    # # print(img_data.eval())
    # result = tf.unstack(result, axis=0)

    # 随机区域框，包含标注框标注区域的0.4
    for _ in range(10):
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(img_data),
                                                                            bounding_boxes=boxes,
                                                                            min_object_covered=1,
                                                                            area_range=[0.1,0.2])
        distorted_image = tf.slice(img_data, begin, size)
        # print(distorted_image.eval().shape)

    # 按照标注框截取图片
    # offset_height = int(xml['ymin'])
    # offset_width = int(xml['xmin'])
    # target_height = int(xml['ymax']) - offset_height
    # target_width = int(xml['xmax']) - offset_width
    # crop_img = tf.image.crop_to_bounding_box(img_data, offset_height, offset_width, target_height, target_width)
    # np_img = sess.run(crop_img)
    #
    # mp.subplot(151)
    # mp.imshow(np_img.astype('float32'))
    #
    # batch_crop_img = tf.expand_dims(crop_img, 0)

# 测试使用何种resize方式，选用resize_area（250，250）
    #
    #
    # img_area = tf.image.resize_area(batch_crop_img, (250, 250))
    # img_area = tf.unstack(img_area, axis=0)[0]
    # np_area = sess.run(img_area)
    # mp.subplot(152)
    # mp.imshow(np_area)
    #
    # img_area = tf.image.resize_bicubic(batch_crop_img, (250, 250))
    # img_area = tf.unstack(img_area, axis=0)[0]
    # np_area = sess.run(img_area)
    # mp.subplot(153)
    # mp.imshow(np_area)
    #
    # img_area = tf.image.resize_bilinear(batch_crop_img, (250, 250))
    # img_area = tf.unstack(img_area, axis=0)[0]
    # np_area = sess.run(img_area)
    # mp.subplot(154)
    # mp.imshow(np_area)
    #
    # img_area = tf.image.resize_nearest_neighbor(batch_crop_img, (250, 250))
    # img_area = tf.unstack(img_area, axis=0)[0]
    # np_area = sess.run(img_area)
    # mp.subplot(155)
    # mp.imshow(np_area)
    # mp.show()


    # show
    # mp.subplot(121)
    # mp.imshow(distorted_image.eval())
    #
    # mp.subplot(122)
    # img_data = tf.expand_dims(img_data, 0)
    # result = tf.image.draw_bounding_boxes(img_data, bbox_for_draw)
    # result = tf.unstack(result, axis=0)
    # mp.imshow(result[0].eval())
    #
    # mp.show()


##