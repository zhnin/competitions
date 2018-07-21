# @Time    : 2018/7/21 17:07
# @Author  : cap
# @FileName: image_preprocess_2.py
# @Software: PyCharm Community Edition
# @introduction:

import tensorflow as tf

import find_jpg


test_list_files, train_dict_file, xmls = find_jpg.get_file_names('D:\\softfiles\\workspace\\games\\xuelang')
