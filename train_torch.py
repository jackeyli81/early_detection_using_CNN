from __future__ import print_function

import argparse, csv, os

import numpy as np, torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#   http://europepmc.org/abstract/MED/26140652
import models

from utils import progress_bar, check_location
from common import config
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIB_DEVI


for pre_parse in [0]:
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model', default="torch_model", type=str,
                        help='model type (default: torch_model)')
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epoch', default=160, type=int,
                        help='total epochs to run')
    # parser.add_argument('--no-augment', dest='augment', action='store_false',
    #                     help='use standard augmentation (default: True)')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    # parser.add_argument('--alpha', default=1., type=float,
    #                     help='mixup interpolation coefficient (default: 1)')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
if not use_cuda:
    raise Exception('what hell u are doing')
best_acc = 0                # best test accuracy
start_epoch = 0             # start from epoch 0 or last checkpoint epoch
if args.seed != 0:
    torch.manual_seed(4550)




print('==> Preparing data..')
from dataset_torch import dataset_hos6
for pre_dataset in [0]:
    TMP1 = np.float32(np.load(config.np_X) )
    TMP2 = np.float32(np.load(config.np_X_te) )
    TMP = np.vstack((TMP1, TMP2))
    
    # print(TMP.shape)
    # raise Exception('Test line 60')
    
    label = np.hstack( (np.load(config.np_y), np.load(config.np_y_te)) )
    from random import sample
    N = TMP1.shape[0] + TMP2.shape[0]
    tr_indice = sample(list(range(N)), TMP1.shape[0])
    te_indice = list( set(range(N)).difference(set(tr_indice)) )
    TMP1 = TMP[tr_indice,:,:,:]
    y1 = label[tr_indice]
    TMP2 = TMP[te_indice,:,:,:]
    y2 = label[te_indice]
    del(TMP, tr_indice, te_indice, N, label)
    
    
    
    TMP1 = np.transpose(TMP1, axes = (0,3,1,2) )
    trainset = dataset_hos6(data_tensor = torch.Tensor( TMP1 ).repeat(4,1,1,1),
                            target_tensor = torch.Tensor(np.float32( y1 ) ).repeat(4).long(),
                            ds_name='train' )
    del(TMP1)
    # print('line 52: ', trainset.shap_())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, drop_last=True)

    
    TMP2 = np.transpose(TMP2, axes = (0,3,1,2) )
    testset = dataset_hos6(data_tensor = torch.Tensor( TMP2 ),
                            target_tensor = torch.Tensor(np.float32( y2  ) ).long() )
    del(TMP2)
    # print('line 59: ', testset.shap_())
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=8)





# Model Configuration
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()







check_location('./results')
logname = ('./results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')
if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')



# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)

# def executer(message = 'line', obj = None, err = False):
#     print(message)
#     print('shape:', obj.shape)
#     print('type', type(obj))
#     if err:
#         raise Exception('your orders.')




def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print(inputs.shape)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # executer('line 130', targets.reshape( (args.batch_size,1) ), False)
        # one_hot = torch.zeros(targets.shape[0], 2).cuda().scatter_(1, targets.reshape( (-1,1) ), 1)
        outputs = net(inputs)#.squeeze()
        # executer('line 135', outputs)
        # one_hot = torch.zeros(targets.shape[0], 2).cuda().scatter_(1, targets.reshape( (-1,1) ), 1)
        loss = criterion(outputs, targets.long() )
        # print(dir(loss))
        # raise Exception('line 130')
        train_loss += loss.cpu().item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            print(inputs.shape)
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.cpu().item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total,
                            correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoints(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)



def checkpoints(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))




def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 40:
        lr /= 2
    if epoch >= 80:
        lr /= 2
    if epoch >= 100:
        lr /= 1.25
    if epoch >= 120:
        lr /= 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



#   主循环。
for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])









#################### depracated #############
'''
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    #Returns mixed inputs, pairs of targets, and lambda
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
'''