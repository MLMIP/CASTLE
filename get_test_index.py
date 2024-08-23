import os
import pickle
import random

import torch

# 设置随机种子
random.seed(42)

def get_unique_prefixes(directory):
    from collections import defaultdict

    # 创建一个字典，键为组合，值为文件名列表
    combination_to_files = defaultdict(list)

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filename = filename.split('.')[0]
            parts = filename.split('_')
            combination = tuple(parts[i] for i in [1, 3, 4, 6] if i < len(parts))
            # 将文件名添加到对应的组合列表中
            combination_to_files[combination].append(filename)

    # 从每个组合列表中随机选择一个文件名
    prefixes = []
    for files in combination_to_files.values():
        if files:  # 确保列表不为空
            min_file = min(files, key=lambda f: int(f.split('_')[2]))
            prefixes.append(min_file)

    return prefixes

def save_prefixes_to_pkl(prefixes, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(prefixes, f)

# 使用示例
directory = 'Data/test_data/res_node_fea'

# 获取唯一文件名前缀集合
prefixes = get_unique_prefixes(directory)
prefixes = list(prefixes)


i =0
alltest_label_list = []
for item in prefixes:
    # 定义PKL文件的路径
    pkl_file_path = 'Data/test_data/label/' + item + '.pt'
    # 读取PKL文件
    data = torch.load(pkl_file_path)
    if data== 0:
        i += 1
    alltest_label_list.append(data)

# 保存前缀集合到pkl文件
save_prefixes_to_pkl(prefixes, 'Data/test_data/data_index/Test.pkl')
save_prefixes_to_pkl(alltest_label_list, 'Data/test_data/data_label_index/Test_label.pkl')



