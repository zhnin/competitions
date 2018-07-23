# @Time    : 2018/7/21 9:47
# @Author  : cap
# @FileName: test.py
# @Software: PyCharm Community Edition
# @introduction:

# 图像压缩test
# 1.37M
# import find_jpg
# test_list_files, train_dict_file, xmls = find_jpg.get_file_names('D:\\softfiles\\workspace\\games\\xuelang')
# xml = xmls['J01_2018.06.13 13_51_09']
# print(len(xmls))
# print(xml)

path= 'D:\\softfiles\\workspace\\games\\xuelang\\xuelang_round1_train_part1_20180628\\正常\\J01_2018.06.13 13_51_09.jpg'

##
import tensorflow as tf



# with open('test.jpg', 'wb') as f:
#     f.write(file)
#     img_raw = tf.image.convert_image_dtype(img_raw, tf.uint8)
#     img_raw = tf.image.encode_jpeg(img_raw, format='rgb', quality=50)
#     with tf.Session() as sess:
#         f.write(img_raw.eval())


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def en_record():
    file = tf.gfile.GFile(path, 'rb').read()
    img_raw = tf.image.decode_jpeg(file, ratio=8, dct_method='INTEGER_FAST', channels=3)

    img_raw = tf.image.convert_image_dtype(img_raw, tf.float32)

    writer = tf.python_io.TFRecordWriter('./test.tfrecord')
    # with tf.Session() as sess:
    #     label = 0
    #     pixes = img_raw.eval().shape
    #     height = pixes[0]
    #     width = pixes[1]
    #     img_raw = tf.image.convert_image_dtype(img_raw, tf.uint8)
    #     img_raw = img_raw.eval().tostring()
    #     # tf.image.resize_area(img_raw)
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         'height': _int64_feature(height),
    #         'width': _int64_feature(width),
    #         'image_raw': _bytes_feature(img_raw)
    #     }))
    #     writer.write(example.SerializeToString())
    # writer.close()

    with tf.Session() as sess:
        label = 0
        pixes = img_raw.eval().shape
        height = pixes[0]
        width = pixes[1]
        img_raw = tf.image.convert_image_dtype(img_raw, tf.uint8)
        img_raw = tf.image.encode_jpeg(img_raw, format='rgb', quality=50)
        # tf.image.resize_area(img_raw)
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(img_raw.eval())
        }))
        writer.write(example.SerializeToString())
    writer.close()

#
def de_record():
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(['D:\\softfiles\\workspace\\git\\competitions\\tianchi\\xuelang\\test\\test.tfrecord'])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(image.eval())
        image, height, width, label = sess.run([image, height, width, label])
        print(len(image))
        import matplotlib.pyplot as mp
        mp.imshow(image)
        mp.show()


if __name__ == '__main__':
    de_record()