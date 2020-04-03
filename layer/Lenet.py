# --*-- coding:utf-8 --*--

"""LeNet in Pytorch"""

import torch.nn as nn

"""
C1: 32*32, kernel:5*5, stride:1, kernel_num:6, feature_map:28*28
M2: 28*28, kernel:2*2, stride:2, kernel_num:6, feature_map:14*14
C3: 14*14, kernel:5*5, stride:1, kernel_num:16, feature_map:10*10
M4: 10*10, kernle:2*2, stride:2, kernel_num:16, feature_map:5*5
C5: 5*5, kernel:5*5, stride:1, kernel_num:120, feature_map:1*1
FC6:120 -> 84
FC7:84 -> 10
"""

cfg = {
    "lenet": [6, "M", 16, "M", 120]
}


class Lenet(nn.Module):
    def __init__(self, lenet_name):
        super(Lenet, self).__init__()
        self.features = self._make_layers(cfg[lenet_name])
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.features(x)
        out = self.fc6(out)
        out = nn.ReLU(inplace=True)
        out = self.fc7(out)
        return out

    def _make_layers(self, cfg_name):
        layers = []
        in_channels = 3
        for x in cfg_name:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 6:
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=5, stride=1, padding=2, bias=True),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
            else:
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=5, stride=1, padding=0, bias=True),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
#         self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
#         self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
#
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = F.relu(out)
#         out = self.max_pool_1(out)
#         out = self.conv2(out)
#         out = F.relu(out)
#         out = self.max_pool_2(out)  # out.shape [30, 16, 5, 5]
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = F.relu(out)
#         out = self.fc2(out)
#         out = F.relu(out)
#         out = self.fc3(out)
#         return out

net = Lenet("lenet")
print("net: ", net)
