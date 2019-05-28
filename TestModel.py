import glob
import time, random

from torch import nn
import numpy as np
import torch
import visdom
import pandas as pd

from model import ResFT
from data import get_data_loader

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = True
BATCH_SIZE = 100


def test(model, dataloader):
    dataloader, _, data_df = dataloader
    each_data_number = data_df.loc['SUM'].values
    model.eval()
    n_correct = 0

    predict_matrix = np.zeros((6, 6))

    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output, _, _ = model(t_img)
            pred = torch.max(class_output.data, 1)
            label = torch.max(t_label, 1)
            n_correct += (pred[1] == label[1]).sum().item()
            for result, answer in zip(pred[1], label[1]):
                if result != answer:
                    predict_matrix[answer][result] += 1

    df = pd.DataFrame(predict_matrix, index=['pd', 'lf', 'hand_expend',
                                    'away_radar', 'forward_radar', 'hand_hold'],
                      columns=['pd', 'lf', 'hand_expend',
                               'away_radar', 'forward_radar', 'hand_hold'])

    erro_sum = df.sum(axis=1).values
    df['Erro SUM'] = erro_sum
    df['Data Total'] = each_data_number
    df['Erro rate'] = erro_sum / each_data_number
    df.loc['SUM'] = df.sum(axis=0).values
    print(df)
    accu = float(n_correct) / len(dataloader.dataset) * 100
    print('accuracy dataset: {:.4f}%'.format(accu))
    return accu, (each_data_number - erro_sum) / each_data_number


def main():
    domain_1_list = [r'I:\Zak_work\State of art\time_freq\time_freq_split_12\RDM\zgy']

    domain_2_list = [r'I:\Zak_work\State of art\zgyclean527\all_people\zk']

    domain_3_list = [r'I:\Zak_work\State of art\zgyclean527\all_people\lt']

    domain_4_list = [r'I:\Zak_work\State of art\zgyclean527\all_people\dl']

    domain_5_list = [r'I:\Zak_work\State of art\zgyclean527\all_people\jc']

    domain_6_list = [r'I:\Zak_work\State of art\zgyclean527\all_people\lg']

    data_array_zgy, label_array_zgy, dfs1 = get_data_loader(BATCH_SIZE, *domain_1_list)
    data_array_zk, label_array_zk, dfs2 = get_data_loader(BATCH_SIZE, *domain_2_list)
    data_array_lt, label_array_lt, dfs3 = get_data_loader(BATCH_SIZE, *domain_3_list)
    data_array_yh, label_array_yh, dfs4 = get_data_loader(BATCH_SIZE, *domain_4_list)
    data_array_jc, label_array_jc, dfs5 = get_data_loader(BATCH_SIZE, *domain_5_list)
    data_array_lg, label_array_lg, dfs6 = get_data_loader(BATCH_SIZE, *domain_6_list)

    data_loader = {'dataloader_zgy': [data_array_zgy, label_array_zgy, dfs1],
                   'dataloader_zk': [data_array_zk, label_array_zk, dfs2],
                   'dataloader_lt': [data_array_lt, label_array_lt, dfs3],
                   'dataloader_yh': [data_array_yh, label_array_yh, dfs4],
                   'dataloader_jc': [data_array_jc, label_array_jc, dfs5],
                   'dataloader_lg': [data_array_lg, label_array_lg, dfs6]}

    # Define model
    model = ResFT().to(DEVICE).double()

    # model dir list
    # path = r'I:\Zak_work\torch_save\ex-05271648\*.pth'
    # files = glob.glob(path)
    files = [r'I:\Zak_work\torch_save\ex-05271648\epoch_48-ylt84.93333333333334-zk95.0.pth',
             ]
    # load model
    print('Cleaned Data.')
    acc_array = []
    for file in files:
        model.load_state_dict(torch.load(file))
        acc_src, acc_m1 = test(model, data_loader.get('dataloader_zk'))
        acc_ylt, acc_m2 = test(model, data_loader.get('dataloader_lt'))
        acc_yh, acc_m3 = test(model, data_loader.get('dataloader_yh'))
        acc_jc, acc_m4 = test(model, data_loader.get('dataloader_jc'))
        acc_lg, acc_m5 = test(model, data_loader.get('dataloader_lg'))
        acc_array.append([acc_src, acc_ylt, acc_yh, acc_lg, acc_jc])
    print(acc_array)


if __name__ == '__main__':
    main()
