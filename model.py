import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random



SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        print(self.conv1.bias.data)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class cifarNet(nn.Module):
    def __init__(self):
        super(cifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(F.max_pool2d(x, 2))
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(F.max_pool2d(x, 2))
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        return x


class simplifiedCifarNet(nn.Module):
    def __init__(self, l_dim):
        super(simplifiedCifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, l_dim[0][0], kernel_size=3)
        self.conv2 = nn.Conv2d(l_dim[1][1], l_dim[1][0], kernel_size=3)
        self.conv3 = nn.Conv2d(l_dim[2][1], l_dim[2][0], kernel_size=3)
        self.fc1 = nn.Linear(l_dim[3][1], l_dim[3][0])
        self.fc2 = nn.Linear(l_dim[4][1], 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv3(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
