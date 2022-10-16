import os
import sys
import random
import argparse
import numpy as np
import scipy.io as scio
import h5py

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model1D import *
from utils import Logger
import random


def load_hdf5(infile, keys):
    with h5py.File(infile, 'r') as f:
        return {key: f[key][:] for key in keys}


def load_mat(infile, keys):
    d_new = scio.loadmat(infile)
    return {key: d_new[key][:] for key in keys}


def read_data(file):
    # data = load_mat(file, {'data'})
    data = load_hdf5(file, {'data'})
    print(data.keys(), data['data'].shape)
    n_samples, n_bases, n_antennas, n_subcarriers = data['data'].shape  # (N, 4, 4, 128)
    d = data['data']  # (n_samples, n_bases, 4, 128)
    d = np.transpose(d, (0, 1, 3, 2))
    np_data = d.reshape((n_samples * n_bases, n_antennas, n_subcarriers))  # (n_samples*n_bases, 4, 128)
    print(np_data.shape, np_data[0, 0, :5])

    if file.find('NLOS') > 0:
        label = np.zeros((n_samples, 1), dtype=int)
    else:
        label = np.ones((n_samples, 1), dtype=int)
    label_trans = np.repeat(label, repeats=[n_bases, ], axis=1)
    np_label = label_trans.reshape(n_samples * n_bases,)
    print(np_label.shape, np_label[:5])

    return np_data, np_label


def prepare_traindata(scene):
    np_pos_data, np_pos_label = read_data('../simdata/'+scene+'/CIR_train_LOS_128.mat')
    np_neg_data, np_neg_label = read_data('../simdata/'+scene+'/CIR_train_NLOS_128.mat')
    data = np.concatenate((np_pos_data, np_neg_data), axis=0)
    label = np.concatenate((np_pos_label, np_neg_label), axis=0)
    return data, label


def prepare_testdata(scene):
    np_pos_data, np_pos_label = read_data('../simdata/'+scene+'/CIR_test_LOS_128.mat')
    np_neg_data, np_neg_label = read_data('../simdata/'+scene+'/CIR_test_NLOS_128.mat')
    data = np.concatenate((np_pos_data, np_neg_data), axis=0)
    label = np.concatenate((np_pos_label, np_neg_label), axis=0)
    return data, label


class DataPreProcess(nn.Module):
    def __init__(self, ndata):
        super(DataPreProcess, self).__init__()
        self.ndata = ndata

    def forward(self, x):
        # x: numpy data with size (n_samples*n_bases, 4, 128)
        data_trans = x #* 1.0E6

        return data_trans


class TrainDataFeeder(Dataset):
    def __init__(self, _data, _label):
        super(TrainDataFeeder, self).__init__()

        ndata = _data.shape[0]   # (n_samples*n_bases, 4, 128)
        dpp = DataPreProcess(ndata)
        _data_smooth = dpp(_data)  # (n_samples*n_bases, 4, 128)

        # print(_data_smooth.shape)
        self.data, self.label = _data_smooth, _label

    def __getitem__(self, index):
        data = self.data[index, :, :]  #
        label = self.label[index]

        data = torch.tensor(data, dtype=torch.float32)
        input_data = torch.squeeze(data)
        input_label = torch.tensor(label, dtype=torch.long)

        return input_data, input_label

    def __len__(self):
        return self.data.shape[0]


def train(idx, net, train_loader, test_loader, cfg):
    # Define Loss
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # Define Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    # Define Sheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epoch)

    best_acc = 0
    epoch_accs = []
    for epoch in range(cfg.epoch):
        net.train()
        total_loss = 0
        nsamples = 0
        epoch_correct = 0
        for i, (data, label) in enumerate(train_loader):
            inputs, labels = data.cuda(), label.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            epoch_correct += predicted.eq(labels.data).sum()

            total_loss += loss
            nsamples += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_acc = 100. * float(epoch_correct) / nsamples
        print('Cycle %d, Epoch %d, Train_Loss = %6.6f, lr = %.6f, Train_Acc = %6.2f%%' % (
            idx+1, epoch, total_loss/nsamples, optimizer.param_groups[0]['lr'], epoch_acc))

        net.eval()
        nsamples = 0
        epoch_correct = 0
        for i, (data, label) in enumerate(test_loader):
            inputs, labels = data.cuda(), label.cuda()

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            epoch_correct += predicted.eq(labels).sum()

            nsamples += labels.size(0)

        epoch_acc = 100. * float(epoch_correct) / nsamples
        print('Cycle %d, Epoch %d, Test_Acc = %6.2f%%' % (idx+1, epoch, epoch_acc))

        if best_acc < epoch_acc:
            best_acc = epoch_acc
            mpath = os.path.join(cfg.exp_dir, 'model' + str(idx+1) + '_best.pkl')
            torch.save(net, mpath)

        if (epoch + 1) % cfg.epoch == 0:
            mpath = os.path.join(cfg.exp_dir, 'model' + str(idx+1) + '_epoch' + str(epoch+1) + '.pkl')
            torch.save(net, mpath)

        scheduler.step()
        epoch_accs.append(epoch_acc)

    return epoch_accs


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='log_CIR_')
    parser.add_argument('--scene', type=str, default='indoor', help='[indoor, InF_DH]')
    parser.add_argument('--arch', type=str, default='net8', help='[net4, net8, net12, net16]')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--num_works', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.1, help='')
    parser.add_argument('--n_steps', type=int, default=20, help='number of epochs to update learning rate')
    return parser.parse_args()


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    cfg = parser_args()
    cfg.exp_dir = cfg.exp_dir + \
                  '_' + cfg.scene + \
                  '_' + cfg.arch + \
                  '_epoch' + str(cfg.epoch) + \
                  '_bs' + str(cfg.batch_size) + \
                  '_lr' + str(cfg.lr)

    sys.stdout = Logger(os.path.join(cfg.exp_dir, 'log_train.txt'))
    print(cfg)

    all_accs = []
    for i in range(5):
        setup_seed(2020+i*3000)

        train_data, train_label = prepare_traindata(cfg.scene)
        DATA_MEAN = train_data.mean(0).mean(0).mean(0)
        print(DATA_MEAN)
        train_data = (train_data / DATA_MEAN)
        train_set = TrainDataFeeder(train_data, train_label)
        train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_works,
                                  batch_size=cfg.batch_size, shuffle=True, drop_last=True)

        test_data, test_label = prepare_testdata(cfg.scene)
        test_data = (test_data / DATA_MEAN)
        test_set = TrainDataFeeder(test_data, test_label)
        test_loader = DataLoader(dataset=test_set, num_workers=cfg.num_works,
                                 batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        # Define Network
        net = None
        if cfg.arch == 'net4':
            net = Net4().cuda()
        elif cfg.arch == 'net8':
            net = Net8().cuda()
        elif cfg.arch == 'net12':
            net = Net12().cuda()
        elif cfg.arch == 'net16':
            net = Net16().cuda()
        elif cfg.arch == 'mwtcnn':
            net = MWTCNN().cuda()
        else:
            print('arch is error.')

        if net is not None:
            one_acc = train(i, net, train_loader, test_loader, cfg)
            all_accs.append(one_acc)

    for one_acc in all_accs:
        print(one_acc)

    np_accs = np.array(all_accs)
    mean_acc = np_accs.mean(axis=0)
    var_acc = np_accs.var(axis=0)
    print('\nMean:')
    for x in list(mean_acc):
        print('{:.2f}'.format(x), end=',')
    print('\nVar:')
    for x in list(var_acc):
        print('{:.2f}'.format(x), end=',')
