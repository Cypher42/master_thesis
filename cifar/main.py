'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from utils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

import os
import argparse

from matplotlib import pyplot as plt

import csv

import time

from models.vgg import VGG
from models.resnet import ResNet18
from models.googlenet import GoogLeNet
from utils import progress_bar


# Training
def train(epoch, progress_bar):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(batch_idx+1), 100.*correct/total

def test(epoch, progress_bar, exp_name='some_exp'):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds += list(predicted)
            all_targets += list(targets)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+exp_name+'.t7')
        best_acc = acc
    return all_targets, all_preds, test_loss/(batch_idx+1), 100.*correct/total,




if __name__ == '__main__':

    exp_prefix = 'enrichment_2_'
    networks = ['resnet','googlenet', 'vgg']
    #preprocessing = ['vanilla','random_cropping','random_affine','random_property','all']
    preprocessing = ['horizontal_flip','random_affine','all']

    tot = len(networks)*len(preprocessing)
    c = 0
    for cnet in networks:
        for pre in preprocessing:

            c += 1
            if c == 1:
                continue

            exp_name = exp_prefix + cnet + '_' + pre

            print('Processing experiment:',exp_name, c, '/', tot)

            parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
            parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
            parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
            args = parser.parse_args()

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            best_acc = 0  # best test accuracy
            start_epoch = 0  # start from epoch 0 or last checkpoint epoch

            # Data
            print('==> Preparing data..')

            if pre == 'vanilla':
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            elif pre == 'random_cropping':
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            elif pre == 'random_affine':
                transform_train = transforms.Compose([


                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            elif pre == 'random_property':
                transform_train = transforms.Compose([
                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            elif pre == 'horizontal_flip':
                transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            elif pre == 'all':
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                            transforms.RandomAffine(degrees=15),
                                            transforms.RandomAffine(0, translate=(0.15, 0.15)),
                                            transforms.RandomAffine(0, scale=(0.15, 0.15)),
                                            transforms.RandomAffine(0, shear=15),
                                            transforms.RandomHorizontalFlip()])
                    ,
                    transforms.ToTensor()
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=4)

            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            # Model
            print('==> Building model..')
            if cnet == 'vgg':
                net = VGG('VGG19')
            elif cnet == 'resnet':
                net = ResNet18()
            elif cnet == 'googlenet':
                net = GoogLeNet()
            # net = PreActResNet18()
            # net = GoogLeNet()
            # net = DenseNet121()
            # net = ResNeXt29_2x64d()
            # net = MobileNet()
            # net = MobileNetV2()
            # net = DPN92()
            # net = ShuffleNetG2()
            # net = SENet18()
            net = net.to(device)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

            if args.resume:
                # Load checkpoint.
                print('==> Resuming from checkpoint..')
                assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
                checkpoint = torch.load('./checkpoint/'+exp_name+'.t7')
                net.load_state_dict(checkpoint['net'])
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch']

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

            train_loss = []
            train_acc = []

            val_loss = []
            val_acc = []



            for epoch in range(start_epoch, start_epoch+200):
                l, a = train(epoch,progress_bar)
                _, _, tl, ta = test(epoch,progress_bar,exp_name)
                train_loss.append(l)
                train_acc.append(a)
                val_acc.append(ta)
                val_loss.append(tl)

            target, predictions, _, _ = test(21,progress_bar,exp_name)


            cm = confusion_matrix(target,predictions)
            names = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
            plot_confusion_matrix(cm, names,savename='.'+os.sep+'results'+os.sep+'confusion_matrices'+os.sep+exp_name)
            plt.clf()
            plot_confusion_matrix(cm, names,normalize=True,savename='.'+os.sep+'results'+os.sep+'confusion_matrices'+os.sep+exp_name+'_normed')
            plt.clf()
            plt.plot(list(range(len(train_acc))),train_acc, label='training accuracy')
            plt.plot(list(range(len(val_acc))),val_acc, label='validation accuracy')
            plt.ylim((0,100))
            plt.legend()
            plt.savefig('.'+os.sep+'results'+os.sep+'accuracy_plots'+os.sep+exp_name, bbox_inches='tight')
            plt.clf()
            plt.plot(list(range(len(train_acc))),train_loss, label='training loss')
            plt.plot(list(range(len(val_acc))),val_loss, label='validation loss')
            plt.ylim((0,2.0))
            plt.legend()
            plt.savefig('.'+os.sep+'results'+os.sep+'loss_plots'+os.sep+exp_name, bbox_inches='tight')
            plt.clf()

            with open('.'+os.sep+'results'+os.sep+'metric_csv'+os.sep+'train_losses.csv','a+', newline='') as fp:
                wrtr = csv.writer(fp)
                wrtr.writerow([exp_name]+train_loss)
            with open('.'+os.sep+'results'+os.sep+'metric_csv'+os.sep+'val_losses.csv','a+', newline='') as fp:
                wrtr = csv.writer(fp)
                wrtr.writerow([exp_name]+val_loss)
            with open('.'+os.sep+'results'+os.sep+'metric_csv'+os.sep+'train_acc.csv','a+', newline='') as fp:
                wrtr = csv.writer(fp)
                wrtr.writerow([exp_name]+train_acc)
            with open('.'+os.sep+'results'+os.sep+'metric_csv'+os.sep+'val_acc.csv','a+', newline='') as fp:
                wrtr = csv.writer(fp)
                wrtr.writerow([exp_name]+val_acc)




