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
BATCH_SIZE = 100
N_EPOCH = 200
LOG_INTERVAL = 20
LEARNING_RATE = 1e-1
DECAY = LEARNING_RATE / N_EPOCH

result = []


# Folder Create
def create_root(model):
    # time_stamp = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
    time_stamp = 888
    model_root = 'I:\\Zak_work\\torch_model_save\\' + str(time_stamp)

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


def train(model, **data_loader):
    '''

    :param data_loader: a dict like this {'dataloader_src':loader_1, 'dataloader_tar':loader_2,
                                            'dataloader_src_test':loader_3, 'dataloader_tar_test':loader_4}
    '''

    global LEARNING_RATE
    time_stamp = time.strftime('ex-%H%M', time.localtime(time.time()))
    env_name = 'res_fine_tune_res'+time_stamp
    vis = visdom.Visdom(env=env_name)

    vis.text('res_fine_tune_res lr*0.001 lr step 25 === EX-3')
    vis.text('1.Add ylt data to test. '
             '2.Change step from 20 to 25. '
             '3.Add every gesture acc line'
             '========================='
             '5-20:'
             '1. Every 5 epoch output. '
             '2. Add Learning Rate line. '
             '3. Regain step to 20'
             '4. FLR/CLR = 0.001'
             '======================='
             '520yyh '
             '1. Correct LR change. '
             '=======================521 '
             '1. Add yh jc data '
             '2. Add model save ')
    vis.text('Hyper-parameters:'
             '1. Learning rate decay multiple policy with step 25  LearningRate * (1 - epoch / N_epoch) ** power. '
             '2. Epoch 200. ' 
             '3. Feature LR / Classifier LR = 0.01'
             '4. BATCH_SIZE = 100')
    len_dataset = data_loader.get('dataloader_src').dataset.tensors[0].size()[0]

    for epoch in range(1, N_EPOCH + 1):
        # acc_src = test(model, data_loader.get('dataloader_tar'), epoch, data_loader.get('datafram_2'))


        model.train()
        loss_class = torch.nn.CrossEntropyLoss()
        train_loss = []
        # LEARNING_RATE = poly_lr_scheduler(epoch, lr_decay_iter=1, max_iter=N_EPOCH, power=1.5)
        if (epoch + 1) % 25 == 0:
            LEARNING_RATE = LEARNING_RATE * (1 - epoch / N_EPOCH) ** 1.5
            print('Learning Rate :', LEARNING_RATE)

        optimizer = torch.optim.SGD([
            {'params': model.feature.parameters(), 'lr': LEARNING_RATE / 100},
        ], lr=LEARNING_RATE, momentum=0.9, weight_decay=DECAY)

        vis.line(X=[epoch], Y=[LEARNING_RATE], win='Learning rate', opts={'title': 'Learning rate'}, update='append')


        for times, data_src_iter in enumerate(data_loader.get('dataloader_src')):
            # Training model using source data
            s_img, s_label = data_src_iter[0].to(DEVICE), data_src_iter[1].to(DEVICE)

            class_output = model(s_img)
            err = loss_class(class_output, torch.max(s_label, 1)[1])
            train_loss.append(err.item())

            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            if times % 20 == 0:
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

        if epoch % 5 == 0 :

            acc_src, acc_m1 = test(model, data_loader.get('dataloader_tar'), epoch, data_loader.get('datafram_2'))
            acc_ylt, acc_m2 = test(model, data_loader.get('dataloader_ylt'), epoch, data_loader.get('datafram_3'))
            acc_yh, acc_m3 = test(model, data_loader.get('dataloader_yh'), epoch, data_loader.get('datafram_4'))
            acc_jc, acc_m4 = test(model, data_loader.get('dataloader_jc'), epoch, data_loader.get('datafram_5'))


            vis.line(X=[epoch], Y=[acc_src], win='acc src', opts={'title': 'acc src'}, update='append')
            vis.line(X=[epoch], Y=[acc_ylt], win='acc ylt', opts={'title': 'acc ylt'}, update='append')
            vis.line(X=[epoch], Y=[acc_yh], win='acc yh', opts={'title': 'acc yh'}, update='append')
            vis.line(X=[epoch], Y=[acc_jc], win='acc jc', opts={'title': 'acc jc'}, update='append')


            # torch.save(model.module.state_dict(), 'run//epoch_{}.pth'.format(epoch))
        vis.line(X=[epoch], Y=[train_loss], win='train_loss', opts={'title': 'train_loss'}, update='append')
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


if __name__ == '__main__':
    domain_1_list = [r'I:\Zak_work\State of art\time_freq\time_freq_split_12\RDM\zgy']

    domain_2_list = [r'I:\Zak_work\State of art\time_freq\time_freq_split_12\RDM\zk',
                     r'I:\Zak_work\State of art\time_freq\time_freq_split_4ant\RDM\zk']

    domain_3_list = [r'I:\Zak_work\State of art\lt']

    domain_4_list = [r'I:\Zak_work\State of art\yh']

    domain_5_list = [r'I:\Zak_work\State of art\jc']

    setup_seed(20)
    loader_src_train, loader_src_test, dfs1 = get_data_loader(BATCH_SIZE, *domain_1_list)
    loader_tar_train, _, dfs2 = get_data_loader(BATCH_SIZE, *domain_2_list)
    loader_ylt_train, _, dfs3 = get_data_loader(BATCH_SIZE, *domain_3_list)
    loader_yh_train, _, dfs4 = get_data_loader(BATCH_SIZE, *domain_4_list)
    loader_jc_train, _, dfs5 = get_data_loader(BATCH_SIZE, *domain_5_list)

    data_loader = {'dataloader_src': loader_src_train, 'dataloader_tar': loader_tar_train,
                   'dataloader_src_test': loader_src_test, 'dataloader_ylt':loader_ylt_train,
                   'datafram_2':dfs2, 'datafram_3':dfs3, 'datafram_4':dfs4, 'datafram_5':dfs5,
                   'dataloader_yh':loader_yh_train, 'dataloader_jc':loader_jc_train,}

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = ResFT().to(DEVICE).double()

    model = torch.nn.DataParallel(model).module.cuda()

    model.apply(weight_init)
    for param in model.feature.parameters():
        param.requires_grad = True
    get_model_inf(model)

    # optimizer = torch.optim.Adam(model_dann.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.RMSprop(model.parameters(), LEARNING_RATE)

    train(model, **data_loader)
