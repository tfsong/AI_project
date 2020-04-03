# --*-- coding:utf-8 --*--

# --*-- coding:utf-8 --*--

# --*-- coding:utf-8 --*--

"""
resnet-50 in pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_planes)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=self.expansion * out_planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(num_features=self.expansion * out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_planes)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_planes)
        self.conv3 = nn.Conv2d(in_channels=out_planes, out_channels=self.expansion * out_planes, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.expansion * out_planes)

        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=self.expansion * out_planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(num_features=self.expansion * out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(Resnet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)  # [stride, 1,1,1,1]
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Resnet18():
    return Resnet(BasicBlock, [2, 2, 2, 2], num_classes=10)


def Resnet34():
    return Resnet(BasicBlock, [3, 4, 6, 3], num_classes=10)


def Resnet50():
    return Resnet(Bottleneck, [3, 4, 6, 3], num_classes=10)


def Resnet101():
    return Resnet(Bottleneck, [3, 4, 23, 3], num_classes=10)


def Resnet152():
    return Resnet(Bottleneck, [3, 8, 36, 3], num_classes=10)


def test():
    net = Resnet152()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(net)
    print(y.size())


test()


