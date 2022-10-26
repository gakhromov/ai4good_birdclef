import torch.nn as nn
import torch.nn.functional as F
from config import config, cnn_conf
import torch

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 54, 64)
        self.fc2 = nn.Linear(64, config['num_classes'])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        targets = targets.float()
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss


def loss_ce(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def loss_bcefocal(logits, targets):
    bcef = BCEFocalLoss()
    loss = bcef(logits, targets)
    return loss


class CNNModel(nn.Module):
    def __init__(self, in_dim):
        super(CNNModel, self).__init__()
        filts, kerns, strds, dense = cnn_conf['filters'], cnn_conf['kernels'], cnn_conf['strides'], cnn_conf['dense']

        assert len(filts) == len(kerns) and len(strds) == len(kerns)

        # setup the encoders
        self.enc0 = ConvBnPool(in_dim,   filts[0], kerns[0], strid=strds[0], pad=(0,0))
        self.enc1 = ConvBnPool(filts[0], filts[1], kerns[1], strid=strds[1], pad=(0,0))
        self.enc2 = ConvBnPool(filts[1], filts[2], kerns[2], strid=strds[2], pad=(0,0))
        self.enc3 = ConvBnPool(filts[2], filts[3], kerns[3], strid=strds[3], pad=(0,0))
        self.enc4 = ConvBnPool(filts[3], filts[4], kerns[4], strid=strds[4], pad=(0,0))

        # global average pooling
        self.gap = nn.AvgPool2d(kernel_size=(9,7))
        self.flat = nn.Flatten()

        # dropout
        self.drop = nn.Dropout(p=0.2)

        # dense layers
        self.d1 = nn.Linear(in_features=filts[-1], out_features=dense[0])
        self.d2 = nn.Linear(in_features=dense[0], out_features=dense[1])
        self.d3 = nn.Linear(in_features=dense[1], out_features=dense[2])

        # final layers
        self.final = nn.Linear(in_features=dense[2], out_features=config['n_classes'])

    def forward(self,x):
        # encode
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        # pooling
        x = self.gap(x)
        x = self.flat(x)
        # linear layer
        x = self.d1(x)
        x = self.drop(x)
        x = self.d2(x)
        x = self.drop(x)
        x = self.d3(x)

        x = self.final(x)

        return x

class ConvBnPool(nn.Module):
    def __init__(self, in_dim, out_dim, kern, strid, pad=None, activation="relu"):
        super(ConvBnPool, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kern, padding="same")
        self.bn = nn.BatchNorm2d(num_features=out_dim)
        self.pool = nn.MaxPool2d(kernel_size=kern, stride=strid)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x
        

def get_model(model="basic", in_dim=1):
    if model == "basic":
        return DummyModel()
    if model == "cnn":
        return CNNModel(in_dim)