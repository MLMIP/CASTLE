# -- coding: utf-8 --
import os
import pickle
import random
import re
import shutil
from urllib import request

import math
import numpy
import numpy as np
import pandas as pd
import torch
import wget
from Bio import PDB
from propy import AAComposition as AAC
from propy import Autocorrelation as AC
from propy import CTD as CTD
from propy import QuasiSequenceOrder as QSO
from propy import ProCheck as PC
from scipy.sparse import coo_matrix

numpy.random.seed(42)
random.seed(42)


def RemoveDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！

    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

def Removefile(file):
    if os.path.exists(file):
        os.remove(file)


def change_three_to_one(res):
    amino_acid = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'PRO': 'P', 'PHE': 'F', 'TYR': 'Y',
                  'TRP': 'W', 'SER': 'S', 'THR': 'T', 'CYS': 'C', 'MET': 'M', 'ASN': 'N', 'GLN': 'Q', 'ASP': 'D',
                  'GLU': 'E', 'LYS': 'K', 'ARG': 'R', 'HIS': 'H',
                  }
    if res in amino_acid:
        res = amino_acid[res]
        return res
    else:
        return '_'


def clean_pdb(pdb_file,chain_id):
    out_pdb_file = '../Data/temp_pdb/' + 'cleanafter.pdb'
    with open(pdb_file,'r') as f_r,open(out_pdb_file,'w') as f_w:
        for line in f_r:
            info = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27], line[30:38],
                    line[38:46], line[46:54]]
            info = [i.strip() for i in info]
            if info[0] == 'ATOM' or info[0] == 'HETAT':
                f_w.write(line)
            if 'ENDMDL' in line:
                break


def change_one_to_three(res):
    three_letter = {'V': 'VAL', 'I': 'ILE', 'L': 'LEU', 'E': 'GLU', 'Q': 'GLN',
                    'D': 'ASP', 'N': 'ASN', 'H': 'HIS', 'W': 'TRP', 'F': 'PHE', 'Y': 'TYR',
                    'R': 'ARG', 'K': 'LYS', 'S': 'SER', 'T': 'THR', 'M': 'MET', 'A': 'ALA',
                    'G': 'GLY', 'P': 'PRO', 'C': 'CYS'}
    if res in three_letter:
        res = three_letter[res]
        return res
    else:
        return '_'


def is_valid_url(url):
    try:
        request.urlopen(url)
        return True
    except:
        return False

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def find_min_distance(list1, list2):
    min_distance = float('inf')
    for point1 in list1:
        for point2 in list2:
            distance = calculate_distance(point1, point2)
            if distance < min_distance:
                min_distance = distance
    return min_distance

def get_intect_type_dist(pdb,metal_pos_list,mut_pos):
    with open(pdb) as f:
        mut_list = []
        metal_list = []
        for line in f:
            line_list = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27],
                         line[30:38],
                         line[38:46], line[46:54], line[-4:]]
            line_list = [i.strip() for i in line_list]
            # 最后一位是换行符
            if line_list[6] == str(mut_pos):
                coord_mut =np.array(line_list[7:10]).astype(np.float64)
                mut_list.append(coord_mut)
            if line_list[6] in metal_pos_list:
                coord_metal = np.array(line_list[7:10]).astype(np.float64)
                metal_list.append(coord_metal)
        return find_min_distance(mut_list,metal_list)

# 20种氨基酸
resList = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
           "SER", "THR", "TRP", "TYR", "VAL"]

def get_pipeipdb_info(pdbname, metal, pos, from_acid, to_acid , label):
    global set_to_acid
    RemoveDir('../Data/demo')
    tag = False
    metal_nan_tag = False
    if pd.isna(metal):
        metal_nan_tag = True

    for i in range(1, 50):
        toX_tag = False
        if to_acid == 'X':
            # 指定要排除的元素
            excluded_value = change_one_to_three(from_acid)
            # 创建一个新的列表，不包含指定的元素
            filtered_list = [x for x in resList if x != excluded_value]
            to_acid = change_three_to_one(random.choice(filtered_list))
            set_to_acid = to_acid
            toX_tag = True
        try:
            url = 'http://metalpdb.cerm.unifi.it/download?t=pdb&id={}_{}'.format(pdbname.lower(), i)
        except:
            print("pdbname:{} 下载失败".format(pdbname))
            break
        if is_valid_url(url) == False: break
        pdbfile = wget.download(url, out='../Data/demo')
        if os.path.getsize(pdbfile) == 0: break
        parser = PDB.PDBParser(QUIET=True)
        struct = parser.get_structure('PDB', pdbfile)
        model = struct[0]
        metal1 = ''
        flag1 = False
        metal_pos_list = []
        type_tag = False
        pdb_array = []
        with open('../Data/demo/{}_{}.site.pdb'.format(pdbname.lower(), i)) as f:
            for line in f:
                # if line[0:4] == 'ATOM' or line[0:4] == 'HETA':
                if line[0:4] == 'ATOM' or line[0:4] == 'HETA':
                    line_list = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27],
                                 line[30:38],
                                 line[38:46], line[46:54], line[-4:]]
                    line_list = [i.strip() for i in line_list]
                    if line_list[0] == 'ATOM' or line_list[0] == 'HETAT':
                        pdb_array.append(line_list)
            pdb_array = np.array(pdb_array, dtype='str')
            for item in pdb_array:
                if item[0] == 'HETAT' and metal_nan_tag == True and item[-1] in metal_List:
                    if type_tag == False:
                        metal = item[-1]
                        type_tag = True
                        flag1 = True
                    metal_pos_list.append(item[6])
                if item[0] == 'HETAT' and metal_nan_tag == False and item[-1] in metal_List:
                    if type_tag == False and item[-1] == str(metal):
                        metal = str(metal)
                        type_tag = True
                        flag1 = True
                    metal_pos_list.append(item[6])
        flag = False
        from_seq = list()
        to_seq = list()
        chain_name = '_'
        len = 0
        for chain in model:
            for res in chain:
                # if not flag: cow_pos += 1
                amino_acid = res.get_resname()
                res_id = res.get_id()
                orig_pos = str(res_id[1]).strip() + str(res_id[2]).strip()
                one_acid = change_three_to_one(amino_acid)
                if one_acid != '_':
                    from_seq.append(one_acid)
                    to_seq.append(one_acid)
                    len += 1
                if one_acid == from_acid and orig_pos == str(pos).strip():
                    chain_name = chain.get_id()
                    to_seq.pop()
                    to_seq.append(to_acid)
                    flag = True
        pocketSeq = "".join(from_seq)
        pocketSeq_to = "".join(to_seq)
        if toX_tag == True:
            to_acid = 'X'
        if flag1 == True and flag == True:
            # foldx得到突变后的pdb
            # pdb = pdbfile.split('/')[-1]
            clean_pdb(pdbfile,chain_name)
            pdb = '../Data/temp_pdb/cleanafter.pdb'
            # pdb = 'Data/foldx/cleanafter.pdb'
            if not os.path.exists(folder_from+'/{}_{}_{}_{}_{}_{}_{}.pdb'.format(metal,pdbname.lower(),i,pos,from_acid,chain_name,to_acid)):
              os.rename(pdb,folder_from+'/{}_{}_{}_{}_{}_{}_{}.pdb'.format(metal,pdbname.lower(),i,pos,from_acid,chain_name,to_acid))
            end_pdb = folder_from+'/{}_{}_{}_{}_{}_{}_{}.pdb'.format(metal,pdbname.lower(),i,pos,from_acid,chain_name,to_acid)
            tag = True
            if get_intect_type_dist(end_pdb,metal_pos_list,pos)<3:
                intect_type = "Direct"
            else:
                intect_type = "Indirect"
            metal_fea = [metal, intect_type, from_acid, to_acid]
            metal_fea = np.array(metal_fea).reshape(1,-1)


            from sklearn.preprocessing import LabelEncoder, OneHotEncoder
            onehotencoder = OneHotEncoder(sparse_output=False,
                                          categories=[metal_List, intect_list, from_acid_list, to_acid_list])
            onehotencoder_res = OneHotEncoder(sparse_output=False, categories=[to_acid_list])
            metal_fea = onehotencoder.fit_transform(metal_fea)

            y = np.array(label).astype(int)
            y = torch.tensor(y, dtype=torch.int64)
            metal_fea = np.array(metal_fea).flatten().astype(float)
            metal_fea = torch.tensor(metal_fea, dtype=torch.float)

            torch.save(y, labelpath_filder + '/{}_{}_{}_{}_{}_{}_{}.pt'.format(metal, pdbname.lower(), i, pos, from_acid,chain_name, to_acid))
            torch.save(metal_fea, metal_filder + '/{}_{}_{}_{}_{}_{}_{}.pt'.format(metal, pdbname.lower(), i, pos, from_acid, chain_name,to_acid))

            all_pdb_tag.append('{}_{}_{}_{}_{}_{}_{}'.format(metal, pdbname.lower(),i,pos,from_acid,chain_name,to_acid))

            ProteinSequence = pocketSeq
            ProteinSequence_to = pocketSeq_to
            try:
                if (PC.ProteinCheck(ProteinSequence) > 0):
                    # print "Protein A Composition (Percent) - 20"
                    # 举例 A=7/39*100 保留三位小数
                    dicAAC = AAC.CalculateAAComposition(ProteinSequence)
                    dicAAC_to = AAC.CalculateAAComposition(ProteinSequence_to)

                    # print "Protein CTD - All - 147"
                    dicCTD = CTD.CalculateCTD(ProteinSequence)
                    dicCTD_to = CTD.CalculateCTD(ProteinSequence_to)

                    # print "Protein Autocorrelation - All - 720"
                    dicAC = AC.CalculateAutoTotal(ProteinSequence)
                    dicAC_to = AC.CalculateAutoTotal(ProteinSequence_to)

                    # print "Protein Quasi-sequence order descriptors - 100"
                    dicQSO = QSO.GetQuasiSequenceOrder(ProteinSequence)
                    dicQSO_to = QSO.GetQuasiSequenceOrder(ProteinSequence_to)

                    # print "Protein Sequence order coupling number descriptors - 60"
                    dicQSO2 = QSO.GetSequenceOrderCouplingNumberTotal(ProteinSequence)
                    dicQSO2_to = QSO.GetSequenceOrderCouplingNumberTotal(ProteinSequence_to)

                    # dicPAAC = PAAC.CalculatePAAComposition(ProteinSequence)
                    dicAll = dict(
                        list(dicAAC.items()) + list(dicCTD.items()) + list(dicAC.items()) + list(dicQSO.items()) + list(
                            dicQSO2.items()))
                    dicAll_to = dict(
                        list(dicAAC_to.items()) + list(dicCTD_to.items()) + list(dicAC_to.items()) + list(
                            dicQSO_to.items()) + list(
                            dicQSO2_to.items()))

                    # featureNames = list(dicAll.keys())
                    seqfeatures = list(dicAll.values())
                    seqfeatures_to = list(dicAll_to.values())
                    seqfeatures.insert(0,pos)
                    seqfeatures_to.insert(0,pos)

                    seqfeatures = np.array(seqfeatures).astype(float)
                    seqfeatures = torch.tensor(seqfeatures, dtype=torch.float)
                    torch.save(seqfeatures,seq_from_filder + '/{}_{}_{}_{}_{}_{}_{}.pt'.format(metal, pdbname.lower(), i, pos, from_acid,chain_name, to_acid))

                    seqfeatures_to = np.array(seqfeatures_to).astype(float)
                    seqfeatures_to = torch.tensor(seqfeatures_to, dtype=torch.float)
                    torch.save(seqfeatures_to,seq_to_filder + '/{}_{}_{}_{}_{}_{}_{}.pt'.format(metal, pdbname.lower(), i, pos, from_acid,chain_name, to_acid))
            except:
                print("********pdb:{},metal:{},pos:{},fromacid:{},toacid:{},xulie_error".format(pdbname, metal, pos, from_acid,to_acid))
                with open(error_examples_file, 'a') as f:
                    f.write("********pdb:{},metal:{},pos:{},fromacid:{},toacid:{},xulie_error".format(pdbname, metal, pos, from_acid,to_acid) + '\n')

    if not tag:
        print("未找到pdb:{},metal:{},pos:{},acid:{},toacid:{}样本对应的氨基酸".format(pdbname, metal, pos, from_acid,to_acid))
        with open(error_examples_file, 'a') as f:
            f.write("未找到pdb:{},metal:{},pos:{},acid:{},toacid:{}样本对应的氨基酸".format(pdbname, metal, pos, from_acid,to_acid) + '\n')
    return


def get_pipeipdb_info_fu(pdbname, metal, pos, from_acid, to_acid , label):
    global set_to_acid
    RemoveDir('../Data/demo')
    tag = False
    metal_nan_tag = False
    if pd.isna(metal):
        metal_nan_tag = True

    for i in range(1, 50):
        toX_tag = False
        if to_acid == 'X':
            # 指定要排除的元素
            excluded_value = change_one_to_three(from_acid)
            # 创建一个新的列表，不包含指定的元素
            filtered_list = [x for x in resList if x != excluded_value]
            to_acid = change_three_to_one(random.choice(filtered_list))
            set_to_acid = to_acid
            toX_tag = True
        try:
            url = 'http://metalpdb.cerm.unifi.it/download?t=pdb&id={}_{}'.format(pdbname.lower(), i)
        except:
            print("pdbname:{} 下载失败".format(pdbname))
            break
        if is_valid_url(url) == False: break
        pdbfile = wget.download(url, out='Data/demo')
        if os.path.getsize(pdbfile) == 0: break
        parser = PDB.PDBParser(QUIET=True)
        struct = parser.get_structure('PDB', pdbfile)
        model = struct[0]
        flag1 = False
        metal_pos_list = []
        type_tag = False
        pdb_array = []
        with open('../Data/demo/{}_{}.site.pdb'.format(pdbname.lower(), i)) as f:
            for line in f:
                # if line[0:4] == 'ATOM' or line[0:4] == 'HETA':
                if line[0:4] == 'ATOM' or line[0:4] == 'HETA':
                    line_list = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27],
                                 line[30:38],
                                 line[38:46], line[46:54], line[-4:]]
                    line_list = [i.strip() for i in line_list]
                    if line_list[0] == 'ATOM' or line_list[0] == 'HETAT':
                        pdb_array.append(line_list)
            pdb_array = np.array(pdb_array, dtype='str')
            for item in pdb_array:
                if item[0] == 'HETAT' and metal_nan_tag == True and item[-1] in metal_List:
                    if type_tag == False:
                        metal = item[-1]
                        type_tag = True
                        flag1 = True
                    metal_pos_list.append(item[6])
                if item[0] == 'HETAT' and metal_nan_tag == False and item[-1] in metal_List:
                    if type_tag == False and item[-1] == str(metal):
                        metal = str(metal)
                        type_tag = True
                        flag1 = True
                    metal_pos_list.append(item[6])
        flag = False
        from_seq = list()
        to_seq = list()
        chain_name = '_'
        len = 0
        for chain in model:
            for res in chain:
                # if not flag: cow_pos += 1
                amino_acid = res.get_resname()
                res_id = res.get_id()
                orig_pos = str(res_id[1]).strip() + str(res_id[2]).strip()
                one_acid = change_three_to_one(amino_acid)
                if one_acid != '_':
                    from_seq.append(one_acid)
                    to_seq.append(one_acid)
                    len += 1
                if one_acid == from_acid and orig_pos == str(pos).strip():
                    chain_name = chain.get_id()
                    to_seq.pop()
                    to_seq.append(to_acid)
                    flag = True
        pocketSeq = "".join(from_seq)
        pocketSeq_to = "".join(to_seq)
        if toX_tag == True:
            to_acid = 'X'
        if flag1 == True and flag == True:
            clean_pdb(pdbfile,chain_name)
            pdb = '../Data/temp_pdb/cleanafter.pdb'
            if not os.path.exists(folder_from+'/{}_{}_{}_{}_{}_{}_{}_fu.pdb'.format(metal,pdbname.lower(),i,pos,from_acid,chain_name,to_acid)):
              os.rename(pdb,folder_from+'/{}_{}_{}_{}_{}_{}_{}_fu.pdb'.format(metal,pdbname.lower(),i,pos,from_acid,chain_name,to_acid))
            end_pdb = folder_from+'/{}_{}_{}_{}_{}_{}_{}_fu.pdb'.format(metal,pdbname.lower(),i,pos,from_acid,chain_name,to_acid)
            tag = True
            if get_intect_type_dist(end_pdb,metal_pos_list,pos)<3:
                intect_type = "Direct"
            else:
                intect_type = "Indirect"
            metal_fea = [metal, intect_type, from_acid, to_acid]
            metal_fea = np.array(metal_fea).reshape(1,-1)


            from sklearn.preprocessing import LabelEncoder, OneHotEncoder
            onehotencoder = OneHotEncoder(sparse_output=False,
                                          categories=[metal_List, intect_list, from_acid_list, to_acid_list])
            onehotencoder_res = OneHotEncoder(sparse_output=False, categories=[to_acid_list])
            metal_fea = onehotencoder.fit_transform(metal_fea)

            y = np.array(label).astype(int)
            y = torch.tensor(y, dtype=torch.int64)
            metal_fea = np.array(metal_fea).flatten().astype(float)
            metal_fea = torch.tensor(metal_fea, dtype=torch.float)

            torch.save(y, labelpath_filder + '/{}_{}_{}_{}_{}_{}_{}_fu.pt'.format(metal, pdbname.lower(), i, pos, from_acid,chain_name, to_acid))
            torch.save(metal_fea, metal_filder + '/{}_{}_{}_{}_{}_{}_{}_fu.pt'.format(metal, pdbname.lower(), i, pos, from_acid, chain_name,to_acid))

            all_pdb_tag.append('{}_{}_{}_{}_{}_{}_{}_fu'.format(metal, pdbname.lower(),i,pos,from_acid,chain_name,to_acid))

            ProteinSequence = pocketSeq
            ProteinSequence_to = pocketSeq_to
            try:
                if (PC.ProteinCheck(ProteinSequence) > 0):
                    # print "Protein A Composition (Percent) - 20"
                    # 举例 A=7/39*100 保留三位小数
                    dicAAC = AAC.CalculateAAComposition(ProteinSequence)
                    dicAAC_to = AAC.CalculateAAComposition(ProteinSequence_to)

                    # print "Protein CTD - All - 147"
                    dicCTD = CTD.CalculateCTD(ProteinSequence)
                    dicCTD_to = CTD.CalculateCTD(ProteinSequence_to)

                    # print "Protein Autocorrelation - All - 720"
                    dicAC = AC.CalculateAutoTotal(ProteinSequence)
                    dicAC_to = AC.CalculateAutoTotal(ProteinSequence_to)

                    # print "Protein Quasi-sequence order descriptors - 100"
                    dicQSO = QSO.GetQuasiSequenceOrder(ProteinSequence)
                    dicQSO_to = QSO.GetQuasiSequenceOrder(ProteinSequence_to)

                    # print "Protein Sequence order coupling number descriptors - 60"
                    dicQSO2 = QSO.GetSequenceOrderCouplingNumberTotal(ProteinSequence)
                    dicQSO2_to = QSO.GetSequenceOrderCouplingNumberTotal(ProteinSequence_to)

                    # dicPAAC = PAAC.CalculatePAAComposition(ProteinSequence)
                    dicAll = dict(
                        list(dicAAC.items()) + list(dicCTD.items()) + list(dicAC.items()) + list(dicQSO.items()) + list(
                            dicQSO2.items()))
                    dicAll_to = dict(
                        list(dicAAC_to.items()) + list(dicCTD_to.items()) + list(dicAC_to.items()) + list(
                            dicQSO_to.items()) + list(
                            dicQSO2_to.items()))

                    # featureNames = list(dicAll.keys())
                    seqfeatures = list(dicAll.values())
                    seqfeatures_to = list(dicAll_to.values())
                    seqfeatures.insert(0,pos)
                    seqfeatures_to.insert(0,pos)

                    seqfeatures = np.array(seqfeatures).astype(float)
                    seqfeatures = torch.tensor(seqfeatures, dtype=torch.float)
                    torch.save(seqfeatures,seq_from_filder + '/{}_{}_{}_{}_{}_{}_{}_fu.pt'.format(metal, pdbname.lower(), i, pos, from_acid,chain_name, to_acid))

                    seqfeatures_to = np.array(seqfeatures_to).astype(float)
                    seqfeatures_to = torch.tensor(seqfeatures_to, dtype=torch.float)
                    torch.save(seqfeatures_to,seq_to_filder + '/{}_{}_{}_{}_{}_{}_{}_fu.pt'.format(metal, pdbname.lower(), i, pos, from_acid,chain_name, to_acid))

            except:
                print("********pdb:{},metal:{},pos:{},fromacid:{},toacid:{},xulie_error".format(pdbname, metal, pos, from_acid,to_acid))
                with open(error_examples_file, 'a') as f:
                    f.write("********pdb:{},metal:{},pos:{},fromacid:{},toacid:{},xulie_error".format(pdbname, metal, pos, from_acid,to_acid) + '\n')

    if not tag:
        print("未找到pdb:{},metal:{},pos:{},acid:{},to_acid:{}样本对应的氨基酸".format(pdbname, metal, pos, from_acid,to_acid))
        with open(error_examples_file, 'a') as f:
            f.write("未找到pdb:{},metal:{},pos:{},acid:{},to_acid:{}样本对应的氨基酸".format(pdbname, metal, pos, from_acid,to_acid) + '\n')
    return


def get_excel_data(file):
    metal = pd.read_excel(file, usecols=['metal'])
    pdbname = pd.read_excel(file, usecols=['PDB'])
    location = pd.read_excel(file, usecols=['pdb_pos'])
    orig_acid = pd.read_excel(file, usecols=['FromAA'])
    to_acid = pd.read_excel(file, usecols=['ToAA'])
    label =  pd.read_excel(file, usecols=['Label'])
    return metal.values, pdbname.values, location.values, orig_acid.values, to_acid.values, label.values


def get_metal_seq_info(excel_add):
    metal, pdbname, location, orig_acid, to_acid, label = get_excel_data(excel_add)
    for i in range(0, len(metal)):
        if pd.isna(location[i][0]):continue
        if label[i][0]==1:
            get_pipeipdb_info(pdbname[i][0], metal[i][0], int(location[i][0]), orig_acid[i][0], to_acid[i][0],label[i][0])
        else:
            get_pipeipdb_info_fu(pdbname[i][0], metal[i][0], int(location[i][0]), orig_acid[i][0], to_acid[i][0],label[i][0])


from_acid_list = ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H']
to_acid_list = ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H', 'X']
metal_List = ["ZN", "MG", "FE", "CU", "CA", "NA", "AS", "HG", "MN", "K", "SM", "W", "CO", "NI", "AU", "CD", "PB", "Y",
              "SR", "PT"]
intect_list = ["Direct", "Indirect"]

if __name__ == '__main__':

    folder_from = '../Data/alltrain_data/metalpdb'
    seq_from_filder = "../Data/alltrain_data/seq_fea"
    seq_to_filder = "../Data/alltrain_data/seq_to_fea"
    metal_filder = "../Data/alltrain_data/metal_fea"
    labelpath_filder = "../Data/alltrain_data/label"
    data_path = '../Data/alltrain_data/record_data.txt'
    error_examples_file = "../Data/alltrain_data/nofind_metalpdb.txt"
    temp_pdb_filder = "../Data/temp_pdb"

    Removefile(error_examples_file)
    Removefile(data_path)
    RemoveDir(folder_from)
    RemoveDir(seq_from_filder)
    RemoveDir(seq_to_filder)
    RemoveDir(metal_filder)
    RemoveDir(labelpath_filder)
    RemoveDir(temp_pdb_filder)

    all_pdb_tag = []
    get_metal_seq_info('../Data/excel/alltrain.xlsx')
    all_pdb_tag = list(set(all_pdb_tag))
    with open(data_path, 'w') as f:
        for item in all_pdb_tag:
            f.write("%s\n" % item)

