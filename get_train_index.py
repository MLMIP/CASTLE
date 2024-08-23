import os
import pickle
import random

# 设置随机种子
random.seed(42)
all_name = []
def get_unique_prefixes(directory):
    from collections import defaultdict
    # 创建一个字典，键为组合，值为文件名列表
    combination_to_files = defaultdict(list)
    # 从文本文件中读取所有文件名
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filename = filename.split('.')[0]
            all_name.append(filename)
            parts = filename.split('_')
            combination = tuple(parts[i] for i in [0, 1, 3, 4, 6, 7] if i < len(parts))
            # 将文件名添加到对应的组合列表中
            combination_to_files[combination].append(filename)

    prefixes = []
    for files in combination_to_files.values():
        if files:  # 确保列表不为空
            min_file = min(files, key=lambda f: int(f.split('_')[2]))
            prefixes.append(min_file)
    return prefixes


def save_prefixes_to_pkl(prefixes, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(prefixes, f)


directory = 'Data/alltrain_data/res_node_fea'

# 获取唯一文件名前缀集合
prefixes = get_unique_prefixes(directory)
prefixes = list(prefixes)

allZn_fu_list = []
allCa_fu_list = []
allMg_fu_list = []
allZnCaMg_fu_list = []

Zn = []
Ca = []
Mg = []
fu = []
all_3_100_fu = []

x = 0
for item in prefixes:
    tag1 =  item.split('_')[0]
    tag2 =  item.split('_')[-1]
    if tag1 == 'ZN' or tag2=='fu':
        allZn_fu_list.append(item)
    if tag1 == 'CA'or tag2=='fu':
        allCa_fu_list.append(item)
    if tag1 == 'MG'or tag2=='fu':
        allMg_fu_list.append(item)
    if tag2=='fu':
        x += 1

    if tag1 == 'ZN' and tag2 != 'fu':
        Zn.append(item)
    if tag1 == 'CA' and tag2 != 'fu':
        Ca.append(item)
    if tag1 == 'MG' and tag2 != 'fu':
        Mg.append(item)
    if tag2 == 'fu' :
        fu.append(item)

allZn_fu_label_list = []
allCa_fu_label_list = []
allMg_fu_label_list = []
all_3_100_fu_label_list = []

selected_1 = random.sample(Zn, 100)
selected_2 = random.sample(Ca, 100)
selected_3 = random.sample(Mg, 100)
all_3_100_fu = selected_1 + selected_2 + selected_3 + fu

for item in allZn_fu_list:
    if item.split('_')[-1] == 'fu':
        allZn_fu_label_list.append(0)
    else:
        allZn_fu_label_list.append(1)

for item in allCa_fu_list:
    if item.split('_')[-1] == 'fu':
        allCa_fu_label_list.append(0)
    else:
        allCa_fu_label_list.append(1)

for item in allMg_fu_list:
    if item.split('_')[-1] == 'fu':
        allMg_fu_label_list.append(0)
    else:
        allMg_fu_label_list.append(1)

for item in all_3_100_fu:
    if item.split('_')[-1] == 'fu':
        all_3_100_fu_label_list.append(0)
    else:
        all_3_100_fu_label_list.append(1)



save_prefixes_to_pkl(all_3_100_fu, 'Data/alltrain_data/data_index/Mix.pkl')
save_prefixes_to_pkl(all_3_100_fu_label_list, 'Data/alltrain_data/data_label_index/Mix_label.pkl')

save_prefixes_to_pkl(allZn_fu_list, 'Data/alltrain_data/data_index/Zn.pkl')
save_prefixes_to_pkl(allZn_fu_label_list, 'Data/alltrain_data/data_label_index/Zn_label.pkl')

save_prefixes_to_pkl(allCa_fu_list, 'Data/alltrain_data/data_index/Ca.pkl')
save_prefixes_to_pkl(allCa_fu_label_list, 'Data/alltrain_data/data_label_index/Ca_label.pkl')

save_prefixes_to_pkl(allMg_fu_list, 'Data/alltrain_data/data_index/Mg.pkl')
save_prefixes_to_pkl(allMg_fu_label_list, 'Data/alltrain_data/data_label_index/Mg_label.pkl')

