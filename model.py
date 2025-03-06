import torch
import torch.nn as nn

class TEN(nn.Module):
    def __init__(self):
        super(TEN, self).__init__()
        # 第一層：卷積層 1->64，核大小7x7，padding=3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        # 第二層：卷積層 64->32，核大小5x5，padding=2
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        # 第三層：卷積層 32->32，核大小3x3，padding=1
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        # 第四層：卷積層 32->1，核大小3x3，padding=1
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        return out
