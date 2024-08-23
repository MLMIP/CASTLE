import pickle
import random

import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class mydataset(Dataset):
    def __init__(self, pdb_tag,fea_dir):
        imgs = []
        # 路径是自己电脑里所对应的路径
        seq_from_filder = fea_dir+"/seq_fea"
        metal_filder = fea_dir+"/metal_fea"
        seq_to_filder = fea_dir+"/seq_to_fea"

        res_node_from_filder = fea_dir+"/res_node_fea"
        res_edge_from_filder = fea_dir+"/res_edge_fea"
        res_index_from_filder = fea_dir+"/res_index_fea"
        atom_node_from_filder = fea_dir+"/atom_node_fea"
        atom2res_from_filder = fea_dir+"/atom2res_fea"
        node_pos_from_filder = fea_dir+"/node_pos_fea"
        atom_edge_from_filder = fea_dir+"/atom_edge_fea"
        atom_index_from_filder = fea_dir+"/atom_index_fea"
        basic_attn_from_filder = fea_dir+"/basic_attn_peiwei"
        for pdbt in pdb_tag:

            atom_edge_wt_path = f'{atom_edge_from_filder}/{pdbt}.pt'
            atoms_edge_index_wt_path = f'{atom_index_from_filder}/{pdbt}.pt'
            res_node_from_path = f'{res_node_from_filder}/{pdbt}.pt'
            res_index_from_path = f'{res_index_from_filder}/{pdbt}.pt'
            metal_fea_path = f'{metal_filder}/{pdbt}.pt'
            seq_from_path = f'{seq_from_filder}/{pdbt}.pt'
            seq_to_path = f'{seq_to_filder}/{pdbt}.pt'
            res_edge_wt_path = f'{res_edge_from_filder}/{pdbt}.pt'
            atom_node_wt_path = f'{atom_node_from_filder}/{pdbt}.pt'
            atom2res_index_wt_path = f'{atom2res_from_filder}/{pdbt}.pt'
            node_pos_from_path = f'{node_pos_from_filder}/{pdbt}.pt'
            basic_attn_from_path = f'{basic_attn_from_filder}/{pdbt}.pt'


            atom_index_from = torch.load(atoms_edge_index_wt_path)
            atom_edge_from = torch.load(atom_edge_wt_path)
            metal_fea = torch.load(metal_fea_path)
            seq_from = torch.load(seq_from_path)
            seq_to = torch.load(seq_to_path)
            res_node_from = torch.load(res_node_from_path)
            res_index_from = torch.load(res_index_from_path)
            res_edge_from = torch.load(res_edge_wt_path)
            atom_node_from = torch.load(atom_node_wt_path)
            atom2res_from = torch.load(atom2res_index_wt_path)
            node_pos_from = torch.load(node_pos_from_path)
            basic_attn_from = torch.load(basic_attn_from_path)

            imgs.append((pdbt,metal_fea,seq_from,seq_to,res_node_from,res_edge_from,res_index_from,atom_node_from,atom_index_from,atom_edge_from,atom2res_from,node_pos_from,basic_attn_from))
        self.imgs = imgs

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        pdbt,metal_fea,seq_from,seq_to,res_node_from,res_edge_from,res_index_from,atom_node_from,atom_index_from,atom_edge_from,atom2res_from,node_pos_from,basic_attn_from = self.imgs[index]
        return pdbt,metal_fea,seq_from,seq_to,res_node_from,res_edge_from,res_index_from,atom_node_from,atom_index_from,atom_edge_from,atom2res_from,node_pos_from,basic_attn_from


