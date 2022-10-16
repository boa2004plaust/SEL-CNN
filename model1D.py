import torch
import torch.nn as nn


def NormalConV(in_chn, out_chn, kernel=(3, ), stride=(1, ), pad=(1, )):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_chn, out_channels=out_chn, kernel_size=kernel, stride=stride, padding=pad),
        nn.BatchNorm1d(out_chn),
        nn.ReLU(),
   )


def ReductionConV(in_chn, out_chn, kernel=(5, ), stride=(2, ), pad=(2, )):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_chn, out_channels=out_chn, kernel_size=kernel, stride=stride, padding=pad),
        #nn.BatchNorm1d(out_chn),
        nn.ReLU(),
   )


def GlobalPooling():
    return nn.AdaptiveAvgPool1d((1, ))


#################################################################################


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = ReductionConV(in_chn=4, out_chn=16)
        self.conv2 = ReductionConV(in_chn=16, out_chn=32)
        self.conv3 = ReductionConV(in_chn=32, out_chn=64)
        self.conv4 = ReductionConV(in_chn=64, out_chn=128)
        self.gap = GlobalPooling()
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Net8(nn.Module):
    def __init__(self):
        super(Net8, self).__init__()
        self.conv1 = ReductionConV(in_chn=4, out_chn=16)
        self.conv1_1 = NormalConV(in_chn=16, out_chn=16)
        self.conv2 = ReductionConV(in_chn=16, out_chn=32)
        self.conv2_1 = NormalConV(in_chn=32, out_chn=32)
        self.conv3 = ReductionConV(in_chn=32, out_chn=64)
        self.conv3_1 = NormalConV(in_chn=64, out_chn=64)
        self.conv4 = ReductionConV(in_chn=64, out_chn=128)
        self.conv4_1 = NormalConV(in_chn=128, out_chn=128)
        self.gap = GlobalPooling()
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_1(x)
        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Net12(nn.Module):
    def __init__(self):
        super(Net12, self).__init__()
        self.conv1 = ReductionConV(in_chn=4, out_chn=16)
        self.conv1_1 = NormalConV(in_chn=16, out_chn=16)
        self.conv1_2 = NormalConV(in_chn=16, out_chn=16)
        self.conv2 = ReductionConV(in_chn=16, out_chn=32)
        self.conv2_1 = NormalConV(in_chn=32, out_chn=32)
        self.conv2_2 = NormalConV(in_chn=32, out_chn=32)
        self.conv3 = ReductionConV(in_chn=32, out_chn=64)
        self.conv3_1 = NormalConV(in_chn=64, out_chn=64)
        self.conv3_2 = NormalConV(in_chn=64, out_chn=64)
        self.conv4 = ReductionConV(in_chn=64, out_chn=128)
        self.conv4_1 = NormalConV(in_chn=128, out_chn=128)
        self.conv4_2 = NormalConV(in_chn=128, out_chn=128)
        self.gap = GlobalPooling()
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Net16(nn.Module):
    def __init__(self):
        super(Net16, self).__init__()
        self.conv1 = ReductionConV(in_chn=4, out_chn=16)
        self.conv1_1 = NormalConV(in_chn=16, out_chn=16)
        self.conv1_2 = NormalConV(in_chn=16, out_chn=16)
        self.conv1_3 = NormalConV(in_chn=16, out_chn=16)
        self.conv2 = ReductionConV(in_chn=16, out_chn=32)
        self.conv2_1 = NormalConV(in_chn=32, out_chn=32)
        self.conv2_2 = NormalConV(in_chn=32, out_chn=32)
        self.conv2_3 = NormalConV(in_chn=32, out_chn=32)
        self.conv3 = ReductionConV(in_chn=32, out_chn=64)
        self.conv3_1 = NormalConV(in_chn=64, out_chn=64)
        self.conv3_2 = NormalConV(in_chn=64, out_chn=64)
        self.conv3_3 = NormalConV(in_chn=64, out_chn=64)
        self.conv4 = ReductionConV(in_chn=64, out_chn=128)
        self.conv4_1 = NormalConV(in_chn=128, out_chn=128)
        self.conv4_2 = NormalConV(in_chn=128, out_chn=128)
        self.conv4_3 = NormalConV(in_chn=128, out_chn=128)
        self.gap = GlobalPooling()
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MWTCNN(nn.Module):
    def __init__(self):
        super(MWTCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 9, 1, 0)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(64, 128, 3, 1, 0)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*29, 128)
        self.fc2 = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        return x


def compute_FLOPS_PARAMS(x, model):
    import thop
    flops, params = thop.profile(model, inputs=(x,), verbose=False)
    print("FLOPs={:.2f}M".format(flops / 1e6))
    print("params={:.2f}M".format(params / 1e6))


if __name__ == '__main__':
    print('###############Net4################')
    model = Net4()
    x = torch.randn(1, 4, 128)
    compute_FLOPS_PARAMS(x, model)

    print('###############Net8################')
    model = Net8()
    x = torch.randn(1, 4, 128)
    compute_FLOPS_PARAMS(x, model)

    print('###############Net12################')
    model = Net12()
    x = torch.randn(1, 4, 128)
    compute_FLOPS_PARAMS(x, model)

    print('###############Net16################')
    model = Net16()
    x = torch.randn(1, 4, 128)
    compute_FLOPS_PARAMS(x, model)

    print('###############MWTCNN################')
    model = MWTCNN()
    x = torch.randn(1, 4, 128)
    compute_FLOPS_PARAMS(x, model)
