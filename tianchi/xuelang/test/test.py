# @Time    : 2018/7/21 9:47
# @Author  : cap
# @FileName: test.py
# @Software: PyCharm Community Edition
# @introduction:

import pickle
with open('./label_map.pickle', 'rb') as f:
    dic = pickle.load(f)

print(dic)