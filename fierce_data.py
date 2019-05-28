import glob

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.utils.data



def path_to_label(path, category=None):
    if category is None:
        category = {'五指握': 'hand_hold',
                    '朝着雷': 'forward_radar',
                    '远离雷': 'away_radar',
                    '五指张': 'hand_expend',
                    '提起 ': 'lift',
                    '按下 ': 'press_down'}
    file_name = path.split('\\')[-1]
    sort = category.get(file_name[0:3], 'the key not exist')

    if sort == 'the key not exist':
        for key in ['按下', '提起', '前拨', '后拨']:
            if file_name.find(key) != -1:
                sort = category.get(file_name[0:2] + ' ', 'Cannot recognition data!')
                break
    return sort


def load_data(data_path, category=None, path_cat = None):
    '''
    :param save_path: save .bin dir
    :type data_path: object
    :param data_path: .mat dir
    :return:
    '''
    training_data = []
    training_label = []
    category_ = glob.glob(data_path + r'\*')
    # print('Start load data at: %s' % data_path)
    file_out = []

    category_col = ['press_down', 'lift', 'hand_expend', 'away_radar', 'forward_radar', 'hand_hold']
    df = pd.DataFrame(columns=category_col)
    data_sorts_col = np.zeros(6)
    for sorts in category_:
        file_list = glob.glob(sorts + r'\*.mat')
        frames2sort = []
        frames_numbers = 0
        #         print('Load data : %s' % sorts.split('\\')[-1])
        for file in file_list:
            if path_to_label(file, path_cat):
                file_name = file.split('\\')[-1]
                file_information = file_name.split('.')[0]  # '朝着雷达 (2)2'
                frame = file_information[-1]
                data_sorts_col[category_col.index(path_to_label(file, path_cat))] += 1
                data_structure = sio.loadmat(file)
                structure_name = 'P' + frame
                file_out.append(file)
                data_array = data_structure[structure_name]
                training_label.append(one_hot(path_to_label(file, path_cat), category))
                #                 training_data.append(np.array(data_array))
                training_data.append(np.moveaxis(np.array(data_array), -1, 0))
    df.loc[data_path.split('\\')[-1]] = data_sorts_col

    training_data = np.uint8(np.array(training_data))
    training_label = np.array(training_label)
    # print('Data shape:{}. Label shape:{}.\n'.format(training_data.shape, training_label.shape), df, '\n')
    return training_data, training_label, file_out, df


def one_hot(y_, category=None):
    '''
    :param category: as follows
    :param y_:
    :return:
    '''
    if category is None:
        category = {'hand_hold': [0, 0, 0, 0, 0, 1],
                    'forward_radar': [0, 0, 0, 0, 1, 0],
                    'away_radar': [0, 0, 0, 1, 0, 0],
                    'hand_expend': [0, 0, 1, 0, 0, 0],
                    'lift': [0, 1, 0, 0, 0, 0],
                    'press_down': [1, 0, 0, 0, 0, 0]}
    return category.get(y_, 'the key not exist')


def get_data_loader(batch_size, *path_dir_list, gama=0, shuffle=True, num_workers=0, infout=True):
    '''
    Train dataset and test dataset must more than one batch size!

    :param batch_size:
    :param path_dir_list:
    :param gama: The ratio of the total number of test to the data set
    :param shuffle:
    :param num_workers:
    :return:
    '''
    category = {'五指握': 'hand_hold',
                '前拨 ': 'forward_radar',
                '后拨 ': 'away_radar',
                '五指张': 'hand_expend',
                '提起 ': 'lift',
                '按下 ': 'press_down'}

    data_array = []
    label_array = []
    dfs = []
    for path in path_dir_list:
        x_data, y_data, file_list, df = load_data(path, path_cat=category)
        dfs.append(df)
        data_array.extend(x_data)
        label_array.extend(y_data)
    dfs = pd.concat(dfs)
    dfs.loc['SUM'] = dfs.sum(axis=0).values

    data_array = np.moveaxis(np.array(data_array), -1, 2)
    label_array = np.array(label_array)

    if infout:
        print('Get data shape:{}. Label shape:{}.\n'.format(data_array.shape, label_array.shape), dfs)

    if shuffle:
        # shuffle
        permutation = np.random.permutation(data_array.shape[0])
        data_array = data_array[permutation, :, :, :]
        label_array = label_array[permutation, :]

    # make data match batch size
    drop_num = data_array.shape[0] % batch_size
    if drop_num:
        if infout:
            print('- - - To match the batch size drop {} data. - - -'.format(drop_num))
        data_array = data_array[:-drop_num]
        label_array = label_array[:-drop_num]

    return data_array, label_array, dfs


if __name__ == '__main__':
    # data shape (2402, 1, 16, 10, 32)
    domain_1_list = [r'I:\Zak_work\State of art\time_freq\time_freq_split_12\RDM\zgy']

    data_array0, label_array0, dfs0 = get_data_loader(30, *domain_1_list, gama=0)
    pass

