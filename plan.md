# 1. Resnet18 微调

## 1.1 网络结构如下

使用resnet18作为特征提取器，从resnet18 第四个卷积块先进行全局平均池化然后输出，后面接全连接层（fc，relu，bn，fc）

```python
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

```

## 1.2 数据集

使用zgy数据共2400个作为训练集，zk数据共1600个作为测试集。

## 1.3 训练

+ 优化器设置：

特征提取层学习率为其他层的1/100. 使用SGD优化。

```python
optimizer = torch.optim.SGD([
        {'params': model.feature.parameters(), 'lr': LEARNING_RATE / 100},
    ], lr=LEARNING_RATE, momentum=0.9, weight_decay=DECAY)
```

+ 学习率下降策略：阶梯梯度下降 multistep

每进行10次迭代后学习率下降一次

学习率策略可参考博客：[学习率如何影响模型性能](https://www.jianshu.com/p/7d3c6ed7c9f1)

```python
if (epoch+1) % 10  == 0:
    lr = LEARNING_RATE * (1 - epoch / N_EPOCH) ** power
    optimizer.param_groups[0]['lr'] = lr
```

![1557502593543](C:\Users\saber\Documents\agit\Learning\1557502593543.png)

## 1.4 初步结果

测试集准确率：

最大值 94.57143， 最终可稳定于93左右，可见已有网络在当前参数下具有一定的泛化性能。

![acc src](I:\Zak_work\State of art\res_fine_tune\plan.assets\375b6266247b4c.svg)

训练损失：

![train loss](I:\Zak_work\State of art\res_fine_tune\plan.assets\375b57e0635ad6.svg)

![1558159980141](I:\Zak_work\State of art\res_fine_tune\plan.assets\1558159980141.png)

## 1.5 实验规划

目标：充分发掘已有的训练好的网络的优良泛化性能，在保证网络的优良泛化性能的同时，迁移学习特征。

ps: 初步实验未计算预测矩阵，后续实验补充。

- a. 替换resnet输出后的全局平均池化为最大值池化。
- b. 加长学习率变化步长。研究表明开始阶段较大的学习率有助于帮助模型跳出局部最优点，从初步实验看出有模型优化方向的转变，加大学习率下降步长有助于寻找最优全局最小值。
- c. 调整特征提取器（resnet18）与后接分类器的学习率比值。初步实验为0.001，调整为 [0.01, 0.0001, 0.00001]

实验数据集设置： zgy数据集共2400作为训练集，zk、ylt数据共2000作为测试集。

## 1.6 实验结果（待续、、、）



Result Error Matrix

b. ![1558348763245](I:\Zak_work\State of art\res_fine_tune\plan.assets\1558348763245.png)

![1558348994085](I:\Zak_work\State of art\res_fine_tune\plan.assets\1558348994085.png)

![1558358018919](I:\Zak_work\State of art\res_fine_tune\plan.assets\1558358018919.png)

加大了学习率变化步长

+ 1.从zk测试集准确率来看模型收敛于极小值点。
+ 2.zk测试集与ylt测试集随着迭代增加，准确率逐渐接近，推测增大epoch会有一定的效果。

# 2. 无监督领域自适应

## 2.1 目的

​		原始论文实现目标为在两个概率分布相似但不同的数据集，训练一个特征提取器采用诸如对抗训练或者计算MMD距离的方法，使得特征提取器的到的特征向量在两个概率分布相似但不同的数据集的分布匹配。

​		上述方法在论文中应用于无监督学习，实现无标签数据集的自动标注。我在手势识别项目中使用的目标是使特征提取器（卷积层）提取的特征分布更加泛化即利用其领域自适应的方法来加强我们模型的定向泛化性能，使用多人的手势数据集来进行领域适应，获取一个适应性极强的特征提取器。目的是之后在用户使用时可以采集少量数据，使用 1 中的fine-tune 方法进行迁移。

## 2.2 初步结果

![1558158279865](I:\Zak_work\State of art\res_fine_tune\plan.assets\1558158279865.png)

使用zgy作为源数据 使用zk 作为target。

## 2.3 进一步实验将在 1 之后规划进行