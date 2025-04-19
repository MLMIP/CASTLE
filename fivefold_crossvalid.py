# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
from numpy import mean, argsort
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, \
    matthews_corrcoef, f1_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from CASTLE.Model.model import Model_Net
import Dataset as dataset
import openpyxl as op

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU训练需要设置这个
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False

#Zn: 0.35
#Ca: 0.44
#Mg: 0.40
#Mix: 0.41
def binary_focal_loss(gamma=2, alpha=0.41):
    # 将 alpha 和 gamma 转换为张量
    alpha = torch.tensor(alpha, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=torch.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        # 确保 y_true 和 y_pred 是浮点型
        y_true = y_true.float()
        y_pred = y_pred.float()

        # 计算 alpha_t
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)

        # 计算 p_t
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred) + 1e-7  # 加上小常数以避免 log(0)

        # 计算焦点损失
        focal_loss = -alpha_t * torch.pow((1 - p_t), gamma) * torch.log(p_t)

        return torch.mean(focal_loss)

    return binary_focal_loss_fixed

def op_toexcel(data,filename):

    if os.path.exists(filename):
        wb = op.load_workbook(filename)
        ws = wb.worksheets[0]

        ws.append(data)
        wb.save(filename)
    else:
        wb = op.Workbook()
        ws = wb['Sheet']
        ws.append(['MCC', 'ACC', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1', 'FPR', 'FNR',
                  'TN', 'FP', 'FN', 'TP','AUPRC','Threshold'])
        ws.append(data)
        wb.save(filename)

def fcvtest(test_pred, test_label,output_result):
    y_pred = test_pred
    y_true = test_label

    y_pred_new = []
    # 根据 F1_score来调节阈值
    best_f1 = 0
    best_threshold = 0.5
    for threshold in range(1, 100):
        threshold = threshold / 100
        binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
        f1 = metrics.f1_score(y_true, binary_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    for value in y_pred:
        if value < best_threshold:
            y_pred_new.append(0)
        else:
            y_pred_new.append(1)
    y_pred_new = np.array(y_pred_new)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_new).ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred, pos_label=1)
    auprc = metrics.auc(recall, precision)
    thd = best_threshold

    print("Matthews: " + str(matthews_corrcoef(y_true, y_pred_new)))
    print("ACC: ", (tp + tn) / (tp + tn + fp + fn))
    print("AUC: ", roc_auc)
    print('sensitivity/recall:', tp / (tp + fn))
    print('specificity:', tn / (tn + fp))
    print('precision:', tp / (tp + fp))
    print('negative predictive value:', tn / (tn + fn))
    print("F1: " + str(f1_score(y_true, y_pred_new)))
    print('error rate:', fp / (tp + tn + fp + fn))
    print('false positive rate:', fp / (tn + fp))
    print('false negative rate:', fn / (tp + fn))
    print('TN:', tn, 'FP:', fp, 'FN:', fn, 'TP:', tp)
    print('AUPRC: ' + str(auprc))
    print('best_threshold: ' + str(best_threshold))

    mcc = float(format((matthews_corrcoef(y_true, y_pred_new)), '.4f'))
    acc = float(format((tp + tn) / (tp + tn + fp + fn), '.4f'))
    auc = float(format(roc_auc, '.4f'))
    sen = float(format(tp / (tp + fn), '.4f'))
    spe = float(format(tn / (tn + fp), '.4f'))
    pre = float(format(tp / (tp + fp), '.4f'))

    npv = float(format(tn / (tn + fn), '.4f'))
    f1 = float(format(f1_score(y_true, y_pred_new), '.4f'))
    fpr = float(format(fp / (tn + fp), '.4f'))
    fnr = float(format(fn / (tp + fn), '.4f'))
    auprc = float(format(auprc, '.4f'))

    result = mcc, acc, auc, sen, spe, pre, npv, f1, fpr, fnr, tn, fp, fn, tp, auprc, thd
    op_toexcel(result, output_result)

def main():
    parser = argparse.ArgumentParser(description="Use CASTLE to fivefold_crossvalid_ZnCaMg")
    parser.add_argument('-id', '--data-index', dest='data_index', type=str,default='Data/alltrain_data/data_index/Zn.pkl',
                        required=False, help='The file stores information about the samples used for five-fold cross-validation.')
    parser.add_argument('-il', '--data-label-index', dest='data_label_index', type=str,default='Data/alltrain_data/data_label_index/Zn_label.pkl',
                        required=False,help='The file stores label information about the samples used for five-fold cross-validation.')
    parser.add_argument('-or', '--outputdir-result', dest='outputdir_result', type=str,default='Result/all_indicator/fiveFold_crossValidation_Zn.xlsx',
                        required=False, help='The folder of the Zn_indicator.')
    parser.add_argument('-of', '--outputdir-file', dest='outputdir_file', type=str,default='Result/all_file/Zn',
                        required=False, help='The folder of the Zn_result_file.')

    args = parser.parse_args()
    data_index_file = args.data_index
    data_label_index_file = args.data_label_index
    output_result = args.outputdir_result
    outputdir_file = args.outputdir_file
    if os.path.exists(output_result):
        os.remove(output_result)  # 删除文件
    if not os.path.exists(outputdir_file):
        os.makedirs(outputdir_file)
    setup_seed(3407) # 使用你想要的种子数值
    epochs = 500
    lr = 0.00002
    batchsize = 1
    GPU_ID = 'cuda:1'
    devices = torch.device(GPU_ID if torch.cuda.is_available() else 'cpu')
    cv = StratifiedKFold(n_splits=5)

    global fpr_keras
    global tpr_keras
    with open(data_index_file, 'rb') as f:
        all_pdb_tag = pickle.load(f)
    with open(data_label_index_file, 'rb') as f:
        all_label = pickle.load(f)
    all_pdb_tag = np.array(all_pdb_tag)
    all_label = np.array(all_label)
    fold_num = 0

    for train,test in cv.split(all_pdb_tag,all_label):
        fold_num += 1
        model = Model_Net()
        model.to(devices)

        criterion = binary_focal_loss()
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)

        pdb_tag_train = all_pdb_tag[train]
        pdb_label_train =  all_label[train]
        pdb_tag_test = all_pdb_tag[test]
        data_test = dataset.mydataset(pdb_tag_test,'Data/alltrain_data')
        data_loader_test = DataLoader(data_test, batch_size=batchsize, shuffle=True,drop_last=True,num_workers=0)

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        best_model_wts = None
        for train_inx, valid_inx in split.split(pdb_tag_train, pdb_label_train):
            pdb_tag_train = all_pdb_tag[train_inx]
            pdb_tag_valid = all_pdb_tag[valid_inx]
            data_train = dataset.mydataset(pdb_tag_train, 'Data/alltrain_data')
            data_loader_train = DataLoader(data_train, batch_size=batchsize, shuffle=True, drop_last=True,
                                           num_workers=0)
            data_valid = dataset.mydataset(pdb_tag_valid, 'Data/alltrain_data')
            data_loader_valid = DataLoader(data_valid, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=0)

            best_val_loss = float('inf')
            epochs_no_improve = 0
            patience = 40

            for epoch in range(epochs):
                one_epoch_y = []
                one_epoch_predic = []
                one_epoch_loss = 0

                count = 1
                model.train()
                for pdb_tag, metal_fea, seq_from, seq_to, res_node_from, res_edge_from, res_index_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, label in data_loader_train:
                    optimizer.zero_grad()
                    seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from = \
                        seq_from.to(devices), seq_to.to(devices), metal_fea.to(devices), res_node_from.to(devices), \
                            res_index_from.to(devices), res_edge_from.to(devices), atom_node_from.to(
                            devices), atom_index_from.to(devices), atom_edge_from.to(devices), atom2res_from.to(
                            devices), node_pos_from.to(devices), basic_attn_from.to(devices)

                    seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from = \
                        torch.squeeze(seq_from, dim=0), torch.squeeze(seq_to, dim=0), torch.squeeze(metal_fea, dim=0), \
                            torch.squeeze(res_node_from, dim=0), torch.squeeze(res_index_from, dim=0), \
                            torch.squeeze(res_edge_from, dim=0), torch.squeeze(atom_node_from, dim=0), torch.squeeze(
                            atom_index_from, dim=0), torch.squeeze(atom_edge_from, dim=0), torch.squeeze(atom2res_from,dim=0), torch.squeeze(
                            node_pos_from, dim=0), torch.squeeze(basic_attn_from, dim=0)

                    pred_y = model(res_node_from, res_index_from, res_edge_from, atom2res_from,
                                   atom_node_from, atom_index_from, atom_edge_from, node_pos_from, basic_attn_from,
                                   seq_from, seq_to, metal_fea, devices)

                    truth_y = label.to(devices).type(torch.float)
                    loss = criterion(truth_y,pred_y)
                    loss.to(devices)
                    out = pred_y.detach().cpu().numpy()
                    y = truth_y.detach().cpu().numpy()
                    one_epoch_predic.append(out.item())
                    one_epoch_y.append(y)
                    one_epoch_loss += loss.detach().cpu().numpy()
                    loss.backward()
                    count += 1
                    optimizer.step()

                one_epoch_result = (np.array(one_epoch_predic) > 0.5)
                acc = accuracy_score(np.array(one_epoch_y), one_epoch_result)
                auc = roc_auc_score(np.array(one_epoch_y), np.array(one_epoch_predic))
                print("第{}次epoch下训练集的ACC为{},AUC为{}".format(epoch, acc, auc))
                print("第{}次epoch下训练集的Loss为{}".format(epoch, one_epoch_loss))
                scheduler.step()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for pdb_tag, metal_fea, seq_from, seq_to, res_node_from, res_edge_from, res_index_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, label in data_loader_valid:
                        seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from = \
                            seq_from.to(devices), seq_to.to(devices), metal_fea.to(devices), res_node_from.to(devices), \
                                res_index_from.to(devices), res_edge_from.to(devices), atom_node_from.to(
                                devices), atom_index_from.to(devices), atom_edge_from.to(devices), atom2res_from.to(
                                devices), node_pos_from.to(devices), basic_attn_from.to(devices)

                        seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from = \
                            torch.squeeze(seq_from, dim=0), torch.squeeze(seq_to, dim=0), torch.squeeze(metal_fea, dim=0), \
                                torch.squeeze(res_node_from, dim=0), torch.squeeze(res_index_from, dim=0), \
                                torch.squeeze(res_edge_from, dim=0), torch.squeeze(atom_node_from, dim=0), torch.squeeze(
                                atom_index_from, dim=0), torch.squeeze(atom_edge_from, dim=0), torch.squeeze(atom2res_from,
                                                                                                             dim=0), torch.squeeze(
                                node_pos_from, dim=0), torch.squeeze(basic_attn_from, dim=0)

                        pred_y = model(res_node_from, res_index_from, res_edge_from, atom2res_from,
                                       atom_node_from, atom_index_from, atom_edge_from, node_pos_from, basic_attn_from,
                                       seq_from, seq_to, metal_fea, devices)
                        label = label.to(devices).type(torch.float)
                        loss = criterion(label,pred_y)
                        val_loss += loss.item()
                val_loss /= len(data_loader_valid)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break



        test_predict = []
        test_y = []
        model.load_state_dict(best_model_wts)
        model.eval()
        df = pd.DataFrame(columns=['PDB', 'Metalpdb_Tag', 'Chain_ID', 'metal', 'FromAA', 'ToAA', 'Location', 'Label', 'Predict'])
        with torch.no_grad():
            for pdb_tag, metal_fea, seq_from, seq_to, res_node_from, res_edge_from, res_index_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, label in data_loader_test:
                seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from = \
                    seq_from.to(devices), seq_to.to(devices), metal_fea.to(devices), res_node_from.to(devices), \
                        res_index_from.to(devices), res_edge_from.to(devices), atom_node_from.to(
                        devices), atom_index_from.to(devices), atom_edge_from.to(devices), atom2res_from.to(
                        devices), node_pos_from.to(devices), basic_attn_from.to(devices)

                seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from = \
                    torch.squeeze(seq_from, dim=0), torch.squeeze(seq_to, dim=0), torch.squeeze(metal_fea, dim=0), \
                        torch.squeeze(res_node_from, dim=0), torch.squeeze(res_index_from, dim=0), \
                        torch.squeeze(res_edge_from, dim=0), torch.squeeze(atom_node_from, dim=0), torch.squeeze(
                        atom_index_from, dim=0), torch.squeeze(atom_edge_from, dim=0), torch.squeeze(atom2res_from,dim=0), torch.squeeze(
                        node_pos_from, dim=0), torch.squeeze(basic_attn_from, dim=0)

                pred_y = model(res_node_from, res_index_from, res_edge_from, atom2res_from,
                               atom_node_from, atom_index_from, atom_edge_from, node_pos_from, basic_attn_from,
                               seq_from, seq_to, metal_fea, devices)

                label = label.to(devices).type(torch.float)
                out = pred_y.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                pdb_tag = ''.join(pdb_tag)
                df = df._append({'PDB': pdb_tag.split('_')[1], 'Metalpdb_Tag': pdb_tag.split('_')[2],'Metal':pdb_tag.split('_')[0],
                                 'Chain_ID': pdb_tag.split('_')[5], 'FromAA': pdb_tag.split('_')[4],
                                 'ToAA': pdb_tag.split('_')[6], 'Pdb_pos': pdb_tag.split('_')[3],
                                 'Label': mean(label),
                                 'Predict': mean(out)}, ignore_index=True)
                test_predict.append(mean(out))
                test_y.append(mean(label))

        df.to_excel(outputdir_file + '{}_fold.xlsx'.format(fold_num), index=False)
        fcvtest(test_predict, test_y,output_result)
        del model

if __name__ == '__main__':
    main()

