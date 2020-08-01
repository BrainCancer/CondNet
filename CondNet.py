import torch
import torch.nn as nn
import fcn


class ReductionBlock(nn.Module):

    def __init__(self, in_channels):
        super(ReductionBlock, self).__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=(4, 4)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 48, kernel_size=(4, 4)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(48, 64, kernel_size=(4, 4)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 96, kernel_size=(4, 4)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Upsample(scale_factor=64)
        )

    def forward(self, x):
        return self.net(x)

class MainBlock(nn.Module):
    
    def __init__(self, in_channels, kernel_size):
        super(MainBlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.reduction_block = ReductionBlock(self.in_channels)
        self.batch_norm = nn.BatchNorm2d(self.in_channels)
        
        if self.kernel_size % 2 == 0:
            self.zero_pad = nn.ZeroPad2d(padding=(3, 0, 3, 0))
            self.conv_layer = nn.Conv2d(self.in_channels + 96, 64, kernel_size=self.kernel_size, padding=0)
        else:
            self.conv_layer = nn.Conv2d(self.in_channels + 96, 64, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
    def forward(self, x):
        x = self.batch_norm(x)
        reduced_x = self.reduction_block(x)

        
        x = torch.cat((x, reduced_x), dim=1)
        if self.kernel_size % 2 == 0:
            x = self.zero_pad(x)
        x = self.conv_layer(x)
        x_mean = x.mean()
        ones = torch.ones(x.shape)
        if torch.cuda.is_available():
          ones = ones.cuda()
        x = torch.cat((x, x_mean * ones), dim=1)
        return x

class CondNet(nn.Module):

    def __init__(self, in_channels, n_class=3):
        super(CondNet, self).__init__()
        self.in_channels = in_channels
        self.n_class = n_class
        self.net = nn.Sequential(
            MainBlock(self.in_channels, 3),
            MainBlock(128, 4),
            MainBlock(128, 5)
        )

        self.cond_linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

        self.cluster_linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 2)
        )

        self.seg_linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        beta = self.cond_linear(x).squeeze(3)
        clust = self.cluster_linear(x)
        seg = self.seg_linear(x).permute(0, 3, 1, 2)
        return beta, clust, seg