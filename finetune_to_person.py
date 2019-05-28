import os
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
BATCH_SIZE = 12
N_EPOCH = 50
LOG_INTERVAL = 20
LEARNING_RATE = 1e-4
DECAY = LEARNING_RATE / N_EPOCH

result = []


# Folder Create
def create_root(direct):
    model_root = 'I:\\Zak_work\\torch_save\\' + str(direct)

    try:
        os.mkdir(model_root)
    except IOError:
        print('{} has exist!'.format(model_root))
    return model_root


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def show_data(vis, data, win='heatmap_', sleep_time=1):
    for i in range(16):
        vis.heatmap(X=data[1, 0, i, :, :], win='h')
        time.sleep(sleep_time)


def test(model, dataloader, epoch, data_df):
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
    print('Epoch: [{}/{}], accuracy dataset: {:.4f}%'.format(epoch, N_EPOCH, accu))
    return accu, (each_data_number - erro_sum) / each_data_number


def train(model, optimizer, **data_loader):
    '''

    :param data_loader: a dict like this {'dataloader_src':loader_1, 'dataloader_tar':loader_2,
                                            'dataloader_src_test':loader_3, 'dataloader_tar_test':loader_4}
    '''
    time_stamp = time.strftime('ex-%m%d%H%M', time.localtime(time.time()))
    # model_save_dir = create_root(time_stamp)
    env_name = 'res_fine-tune_to_person'+time_stamp
    vis = visdom.Visdom(env=env_name)

    len_dataset = data_loader.get('dataloader_zk')[0].dataset.tensors[0].size()[0]

    for epoch in range(1, N_EPOCH + 1):
        # acc_src = test(model, data_loader.get('dataloader_tar'), epoch, data_loader.get('datafram_2'))

        model.train()
        loss_class = torch.nn.CrossEntropyLoss()
        train_loss = []
        lr = poly_lr_scheduler(optimizer, LEARNING_RATE, epoch, lr_decay_iter=1,
                          max_iter=N_EPOCH, power=1.5)
        vis.line(X=[epoch], Y=[lr], win='Learning rate', opts={'title': 'Learning rate'}, update='append')


        for times, data_src_iter in enumerate(data_loader.get('dataloader_zk')[0]):
            # Training model using source data
            s_img, s_label = data_src_iter[0].to(DEVICE), data_src_iter[1].to(DEVICE)

            class_output = model(s_img)
            err = loss_class(class_output, torch.max(s_label, 1)[1])
            train_loss.append(err.item())

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            pred = torch.max(class_output.data, 1)
            n_correct = (pred[1] == torch.max(s_label, 1)[1]).sum().item()
            batch_acc = (n_correct / BATCH_SIZE) * 100
            print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}\tBatch Acc: {}'.format(
                epoch, times * BATCH_SIZE, len_dataset, err.item(),
                batch_acc))
            vis.line(X=[epoch + times / len_dataset], Y=[err.item()],
                     win='batchloss', opts={'title': 'batch loss'}, update='append')
            vis.line(X=[epoch + times / len_dataset], Y=[batch_acc],
                     win='batchacc', opts={'title': 'batch acc'}, update='append')

        train_loss = np.mean(train_loss)

        acc_src, acc_m1 = test(model, data_loader.get('dataloader_zk')[1], epoch, data_loader.get('dataloader_zk')[2])
        acc_ylt, acc_m2 = test(model, data_loader.get('dataloader_lt')[1], epoch, data_loader.get('dataloader_lt')[2])
        acc_yh, acc_m3 = test(model, data_loader.get('dataloader_yh')[1], epoch, data_loader.get('dataloader_yh')[2])
        acc_jc, acc_m4 = test(model, data_loader.get('dataloader_jc')[1], epoch, data_loader.get('dataloader_jc')[2])
        acc_lg, acc_m5 = test(model, data_loader.get('dataloader_lg')[1], epoch, data_loader.get('dataloader_lg')[2])


        vis.line(X=[epoch], Y=[acc_src], win='acc src', opts={'title': 'acc src'}, update='append')
        vis.line(X=[epoch], Y=[acc_ylt], win='acc ylt', opts={'title': 'acc ylt'}, update='append')
        vis.line(X=[epoch], Y=[acc_yh], win='acc yh', opts={'title': 'acc yh'}, update='append')
        vis.line(X=[epoch], Y=[acc_jc], win='acc jc', opts={'title': 'acc jc'}, update='append')
        vis.line(X=[epoch], Y=[acc_lg], win='acc lg', opts={'title': 'acc lg'}, update='append')



        # vis.line(X=[epoch], Y=[train_loss], win='train_loss', opts={'title': 'train_loss'}, update='append')
        # torch.save(model.state_dict(), model_save_dir + '\\epoch_{}-ylt{}-zk{}.pth'.format(epoch, acc_ylt, acc_src))
        # vis.line(X=[epoch], Y=[acc_m1.reshape(6, 1)], win='acc gesture', opts={'title': 'acc gesture'}, update='append')
        # vis.line(X=[epoch], Y=[acc_m2.reshape(6, 1)], win='acc ylt gesture', opts={'title': 'acc ylt gesture'}, update='append')

    try:
        vis.save(envs=[env_name])
    except :
        vis.save()


# 自定义权值初始化方式
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


def get_model_inf(model):
    print(model)
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("参数和：" + str(l))
        k = k + l
    print("总参数和：" + str(k))


def set_grad(model, grad=True):
    for param in model.parameters():
        param.requires_grad = grad


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""
    if (iter+1) % 30  == 0:
        lr = init_lr * (1 - iter / max_iter) ** power
        optimizer.param_groups[0]['lr'] = lr
        print('Learning Rate :', lr)
    else:
        lr = init_lr
    return lr

if __name__ == '__main__':
    domain_2_list = [r'I:\Zak_work\State of art\zgyclean527\all_people\zk']

    domain_3_list = [r'I:\Zak_work\State of art\zgyclean527\all_people\lt']

    domain_4_list = [r'I:\Zak_work\State of art\zgyclean527\all_people\dl']

    domain_5_list = [r'I:\Zak_work\State of art\zgyclean527\all_people\jc']

    domain_6_list = [r'I:\Zak_work\State of art\zgyclean527\all_people\lg']

    data_array_zk, test_array_zk, dfs2 = get_data_loader(BATCH_SIZE, *domain_2_list, alpha=4, loader_TT_array=True)
    data_array_lt, test_array_lt, dfs3 = get_data_loader(BATCH_SIZE, *domain_3_list, alpha=1, loader_TT_array=True)
    data_array_yh, test_array_yh, dfs4 = get_data_loader(BATCH_SIZE, *domain_4_list, alpha=1, loader_TT_array=True)
    data_array_jc, test_array_jc, dfs5 = get_data_loader(BATCH_SIZE, *domain_5_list, alpha=1, loader_TT_array=True)
    data_array_lg, test_array_lg, dfs6 = get_data_loader(BATCH_SIZE, *domain_6_list, alpha=1, loader_TT_array=True)

    setup_seed(20)

    data_loader = {'dataloader_zk': [data_array_zk, test_array_zk, dfs2],
                   'dataloader_lt': [data_array_lt, test_array_lt, dfs3],
                   'dataloader_yh': [data_array_yh, test_array_yh, dfs4],
                   'dataloader_jc': [data_array_jc, test_array_jc, dfs5],
                   'dataloader_lg': [data_array_lg, test_array_lg, dfs6]}

    model = ResFT().to(DEVICE).double()

    for param in model.feature.parameters():
        param.requires_grad = True

    # load state dict
    # files = [r'I:\Zak_work\torch_save\ex-05271648\epoch_48-ylt84.93333333333334-zk95.0.pth',
    #          r'I:\Zak_work\torch_save\ex-05271648\epoch_100-ylt85.2-zk92.5.pth',
    #          r'I:\Zak_work\torch_save\ex-05271648\epoch_66-ylt86.4-zk94.57142857142857.pth'
    #          ]
    files = [r'I:\Zak_work\torch_save\ex-05271648\epoch_66-ylt86.4-zk94.57142857142857.pth'
             ]

    # for file in files:
    #     model.load_state_dict(torch.load(file))
    #     # optimizer = torch.optim.Adam(model_dann.parameters(), lr=LEARNING_RATE)
    #     optimizer = torch.optim.RMSprop([
    #             {'params': model.feature.parameters(), 'lr': LEARNING_RATE / 100},
    #         ], LEARNING_RATE)
    #     # optimizer = torch.optim.SGD([
    #     #         {'params': model.feature.parameters(), 'lr': LEARNING_RATE / 100},
    #     #     ], lr=LEARNING_RATE, momentum=0.9, weight_decay=DECAY)
    #     train(model, optimizer, **data_loader)

    for file in files:
        model.load_state_dict(torch.load(file))
        acc_src, acc_m1 = test(model, data_loader.get('dataloader_zk')[1], 0, data_loader.get('dataloader_zk')[2])
        acc_ylt, acc_m2 = test(model, data_loader.get('dataloader_lt')[1], 0, data_loader.get('dataloader_lt')[2])
        acc_yh, acc_m3 = test(model, data_loader.get('dataloader_yh')[1], 0, data_loader.get('dataloader_yh')[2])
        acc_jc, acc_m4 = test(model, data_loader.get('dataloader_jc')[1], 0, data_loader.get('dataloader_jc')[2])
        acc_lg, acc_m5 = test(model, data_loader.get('dataloader_lg')[1], 0, data_loader.get('dataloader_lg')[2])
