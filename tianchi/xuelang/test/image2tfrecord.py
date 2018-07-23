# @Time    : 2018/7/21 17:13
# @Author  : cap
# @FileName: image2tfrecord.py
# @Software: PyCharm Community Edition
# @introduction:
"""
1. 获取文件列表：测试文件列表，train文件列表，train文件标注信息
2. 把train文件的label，mapping处理
3. 获取每类label的样本数量，力求每个类别的样本数量相同，对于少的做增量处理
4. 对图像进行翻转，rgb处理
5. 把样本数据保存成TFrecord格式

6. 读取并解析TFRecord文件
7. 构建iterator

8. 模型构建CNN-keras


"""
# import keras.preprocessing as kp

import tensorflow as tf
g = tf.Graph()
a = tf.constant([1,2], name='a')
b = tf.constant([3,4], name='b')

with g.device('/cpu'):
    result = a + b
    with tf.Session() as sess:
        for i in range(100):
            print(sess.run(result))