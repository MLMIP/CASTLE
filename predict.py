import argparse

import pandas as pd
import torch
from numpy import mean

from generate_predict_fea import generate_predict_input
from get_predict_pdb import get_predict_pdb_info
from model import Model_Net
import Dataset_predict as dataset
from torch.utils.data import DataLoader

def predict(data_loader_predict,outfile):
    # 1. 定义模型结构
    model = Model_Net()
    print(model.state_dict().keys())
    # 2. 定义权重文件路径
    weights_path = 'Result/Zn_smile_test.pth'
    # 3. 加载预训练的权重文件
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(weights_path, map_location=device)
    print(checkpoint.keys())
    # 4. 加载权重到模型中
    model.load_state_dict(checkpoint)
    model.to(device)
    df = pd.DataFrame(columns=['PDB', 'Metalpdb_Tag', 'Chain_ID', 'metal', 'FromAA', 'ToAA', 'pdb_pos', 'Predict'])
    for pdb_tag, metal_fea, seq_from, seq_to, res_node_from, res_edge_from, res_index_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from in data_loader_predict:
        seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from = \
            seq_from.to(device), seq_to.to(device), metal_fea.to(device), res_node_from.to(device), \
                res_index_from.to(device), res_edge_from.to(device), atom_node_from.to(device), atom_index_from.to(
                device), atom_edge_from.to(device), atom2res_from.to(device), node_pos_from.to(device), basic_attn_from.to(device)

        seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from = \
            torch.squeeze(seq_from, dim=0), torch.squeeze(seq_to, dim=0), torch.squeeze(metal_fea, dim=0), \
                torch.squeeze(res_node_from, dim=0), torch.squeeze(res_index_from, dim=0), \
                torch.squeeze(res_edge_from, dim=0), torch.squeeze(atom_node_from, dim=0), torch.squeeze(atom_index_from, dim=0), torch.squeeze(
                atom_edge_from, dim=0), torch.squeeze(atom2res_from, dim=0), torch.squeeze(node_pos_from,dim=0), torch.squeeze(basic_attn_from, dim=0)

        pred_y = model(res_node_from, res_index_from, res_edge_from, atom2res_from,atom_node_from, atom_index_from, atom_edge_from, node_pos_from, basic_attn_from,seq_from, seq_to, metal_fea, device)
        pdb_tag = ''.join(pdb_tag)
        out = pred_y.detach().cpu().numpy()
        df = df._append(
            {'PDB': pdb_tag.split('_')[1], 'Metalpdb_Tag': pdb_tag.split('_')[2], 'metal': pdb_tag.split('_')[0],
             'Chain_ID': pdb_tag.split('_')[5], 'FromAA': pdb_tag.split('_')[4],
             'ToAA': pdb_tag.split('_')[6], 'pdb_pos': pdb_tag.split('_')[3],
             'Predict': mean(out)}, ignore_index=True)
    df.to_excel(outfile , index=False)


def main():
    parser = argparse.ArgumentParser(description="Using SMILE to predict the pathogenicity of mutations.")
    parser.add_argument('-i', '--mutant-file', dest='mutant_file', type=str,default='predict_save_data/predict.xlsx',
                        required=False, help='The file storing the information of the mutations.')
    parser.add_argument('-o', '--output-file', dest='outfile', type=str,default='predict_save_data/SMILE_predict_result.xlsx',
                        required=False, help='The path of the result.')
    parser.add_argument('-d', '--feature-dir', dest='dir', type=str,default='predict_save_data',
                        required=False, help='The path to store intermediate features and model inputs.')
    args = parser.parse_args()
    infile = args.mutant_file
    outfile = args.outfile
    feature_dir = args.dir

    pdbname = pd.read_excel(infile, usecols=['PDB']).values
    metal = pd.read_excel(infile, usecols=['Metal']).values
    location = pd.read_excel(infile, usecols=['pdb_pos']).values
    orig_acid = pd.read_excel(infile, usecols=['FromAA']).values
    to_acid = pd.read_excel(infile, usecols=['ToAA']).values
    all_testpdb_tag = []
    for i in range(0, len(pdbname)):
        all_testpdb_tag.append(get_predict_pdb_info(pdbname[i][0],metal[i][0], location[i][0], orig_acid[i][0], to_acid[i][0],feature_dir))

    all_pdb_tag = list(set(all_testpdb_tag))
    with open(feature_dir+'/record_predict_data.txt', 'w') as f:
        for item in all_pdb_tag:
            f.write("%s\n" % item)

    all_predict_tag = generate_predict_input(all_pdb_tag)
    #填预测数据特征及其结果保存的目录
    data_predict = dataset.mydataset(all_predict_tag,feature_dir)
    data_loader_predict = DataLoader(data_predict, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    predict(data_loader_predict,outfile)

if __name__ == "__main__":
    main()