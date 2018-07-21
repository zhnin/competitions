# @Time    : 2018/7/21 10:05
# @Author  : cap
# @FileName: find_jpg.py
# @Software: PyCharm Community Edition
# @introduction: # 查找所有jpg文件并分类，以dic和list的形式
import os
import pickle
import xml.etree.ElementTree as ET

dict_map = {0: '正常', 1: '吊经', 2: '擦洞', 3: '跳花', 4: '毛洞', 5: '织稀', 6: '扎洞',
            7: '缺经', 8: '毛斑', 9: '边扎洞', 10: '缺纬', 11: '油渍', 12: '污渍',
            13: '嵌结', 14: '弓纱', 15: '破边', 16: '边针眼', 17: '吊纬', 18: '回边',
            19: '剪洞', 20: '黄渍', 21: '楞断', 22: '破洞', 23: '粗纱', 24: '织入', 25: '吊弓',
            26: '扎梳', 27: '愣断', 28: '擦伤', 29: '擦毛', 30: '线印', 31: '经粗纱', 32: '经跳花',
            33: '蒸呢印', 34: '边缺纬', 35: '修印', 36: '厚薄段', 37: '扎纱', 38: '毛粒',
            39: '紧纱', 40: '纬粗纱', 41: '结洞', 42: '耳朵', 43: '边白印', 44: '厚段',
            45: '夹码', 46: '明嵌线', 47: '边缺经'}

def get_file_names(path):
    # 获取指定path下的所有训练数据jpg文件
    test_list_files = []
    train_dict_file = {} # key:label ,value:files_list
    for cur_dir, sub_dir, sub_files in os.walk(path):
        # 测试文件名列表
        if cur_dir.split(os.sep)[-1].startswith('xuelang_round1_test_'):
            for test_file in sub_files:
                if test_file.endswith('.jpg'):
                    test_file_path = os.path.join(cur_dir, test_file)
                    test_list_files.append(test_file_path)
        # 训练文件名字典列表
        elif cur_dir.split(os.sep)[-2].startswith('xuelang_round1_train_'):
            # get key
            label = cur_dir.split(os.sep)[-1]
            # label map
            for key, value in dict_map.items():
                if value == label:
                    label = key
                    break
            if label not in train_dict_file.keys():
                train_dict_file[label] = []

            for train_file in sub_files:
                if train_file.endswith('.jpg'):
                    train_file_path = os.path.join(cur_dir, train_file)
                    train_dict_file[label].append(train_file_path)
    # train nums = 2022
    # 获取xml信息{filename:{xml中的信息}}
    xmls = {}
    for label, filepaths in train_dict_file.items():
        for file_path in filepaths:
            xml = {}
            file_path = file_path.replace('.jpg', '.xml')
            if not os.path.isfile(file_path):
                xmls[file_path.split(os.sep)[-1][:-4]] = None
            else:
                root = ET.parse(file_path).getroot()
                for ele in root.iter():
                    xml[ele.tag] = ele.text

                xmls[xml['filename'][:-4]] = xml
    return test_list_files, train_dict_file, xmls


if __name__ == '__main__':
    test_list_files, train_dict_file, xmls = get_file_names('D:\\softfiles\\workspace\\games\\xuelang')
    print(train_dict_file)
    # print(test_list_files, train_dict_file, xmls)
    # 对每个label的数据样本数量进行排序，



