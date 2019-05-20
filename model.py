import torch
import torch.nn as nn
from torchvision import models


class resnet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        # global average pooling to build tail
        tail = nn.MaxPool2d((8, 8))(feature4)
        return feature4, tail


class ResFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = resnet18()
        self.fc1 = torch.nn.Linear(512, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.relu1 = torch.nn.ReLU(True)
        self.fc2 = torch.nn.Linear(512, 6)

    def forward(self, inputs):
        inputs = torch.nn.functional.interpolate(inputs, size=(256, 256), mode='bilinear')
        img, tail = self.feature(inputs)
        X = self.fc1(tail.view((-1, 512)))
        X = self.relu1(X)
        X = self.bn1(X)
        X = self.fc2(X)

        if self.training:
            return X

        return X, img, tail


if __name__ == '__main__':
    model = ResFT()
    model.eval()
    data = torch.Tensor(torch.randn(2, 3, 256, 256))
    out, img, tail = model(data)
    print(out.size(), img.size(), tail.size())


    from data import load_data
    import numpy as np
    import PIL
    import visdom
    import matplotlib.pyplot as plt
    vis = visdom.Visdom(env='main')
    _1_list = r'I:\Zak_work\State of art\time_freq\time_freq_split_12\RDM\zgy'
    category = {'五指握': 'hand_hold',
                '前拨 ': 'forward_radar',
                '后拨 ': 'away_radar',
                '五指张': 'hand_expend',
                '提起 ': 'lift',
                '按下 ': 'press_down'}
    training_data, training_label, file_out, df = load_data(_1_list, path_cat=category)
    # vis.image(training_data[0], win='train', opts={'title': 'train'})

    out, img, tail = model(torch.tensor(training_data[0:6]).float())
    print(out.size(), img.size(), tail.size())

    tmp = training_data[0]
    vis.image(tmp, win='data', opts={'title': '1'})

    # Upsample
    rdm = torch.Tensor(training_data[0:6] * 3.5)

    upsample = torch.nn.functional.interpolate(rdm, size=(256, 256), mode='bilinear')
    vis.images(upsample, win='updata', nrow=3, opts={'title': '2'})
    # vis.image(rdm, win='upsam', opts={'title': 'upsam'})
    # def imshow(tensor, title=None):
    #     image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    #     image = image.squeeze(0)  # remove the fake batch dimension
    #     image = unloader(image)
    #     plt.imshow(image)
    #     if title is not None:
    #         plt.title(title)
    #     plt.pause(0.001)
    print(upsample.size())

