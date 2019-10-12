import os
import argparse
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time

from lenet import LeNet
from mobilenet import MobileNet
from vgg import VGG
from convnet import ConvNet
from resnet import ResNet18, ResNet50

from admm import GradualWarmupScheduler
from admm import CrossEntropyLossMaybeSmooth
from admm import mixup_data, mixup_criterion

from regularizer import L1Regularizer, L2Regularizer, ElasticNetRegularizer, GroupSparseLassoRegularizer, GroupLassoRegularizer




# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--arch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-decay', type=int, default=60, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')

parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='ce mixup')
parser.add_argument('--alpha', type=float, default=0, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


if args.cuda:
    if args.arch == "vgg":
        if args.depth == 16:
            model = VGG(depth=16, init_weights=True, cfg=None)
        elif args.depth == 19:
            model = VGG(depth=19, init_weights=True, cfg=None)
        else:
            sys.exit("vgg doesn't have those depth!")
    elif args.arch == "resnet":
        if args.depth == 18:
            model = ResNet18()
        elif args.depth == 50:
            model = ResNet50()
        else:
            sys.exit("resnet doesn't implement those depth!")
    elif args.arch == "convnet":
        args.depth = 4
        model = ConvNet()
        print("convnet selected")
    elif args.arch == "lenet":
        args.depth = 5
        model = LeNet()
    elif args.arch == "mobilenet":
        args.depth = 13
        model = MobileNet()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
    model.cuda()



#############

criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()
# args.smooth = args.smooth_eps > 0.0
# args.mixup = config.alpha > 0.0

optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

optimizer = None
if(args.optmzr == 'sgd'):
    optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr,momentum=0.9, weight_decay=1e-4)
elif(args.optmzr =='adam'):
    optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)



scheduler = None
if args.lr_scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader), eta_min=4e-08)
elif args.lr_scheduler == 'default':
    # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
    epoch_milestones = [65, 100, 130]

    """Set the learning rate of each parameter group to the initial lr decayed
        by gamma once the number of epoch reaches one of the milestones
    """
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i*len(train_loader) for i in epoch_milestones], gamma=0.5)
else:
    raise Exception("unknown lr scheduler")

if args.warmup:
    scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr/args.warmup_lr, total_iter=args.warmup_epochs*len(train_loader), after_scheduler=scheduler)


#############

def train(train_loader,criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        scheduler.step()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if args.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, args.alpha)

        # compute output
        output = model(input)

        if args.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, args.smooth)
        else:
            ce_loss = criterion(output, target, smooth=args.smooth)
            # l1_reg_loss = L1Regularizer(model=model, lambda_reg=1e-5)
            # ce_loss = l1_reg_loss.regularized_all_param(reg_loss_function=ce_loss)

        # measure accuracy and record loss
        acc1,_ = accuracy(output, target, topk=(1,5))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        ce_loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test(model, criterion, test_loader):
    model.eval()
    losses = AverageMeter()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set loss: {:.4f},  * Acc@1: {}/{} ({:.2f}%)\n'.format(
        losses.avg, correct, len(test_loader.dataset),
        100. * float(correct) / float(len(test_loader.dataset))))
    return losses.avg, (100. * float(correct) / float(len(test_loader.dataset)))

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def main():
    all_acc = [0.000]
    for epoch in range(0, args.epochs):
        # if epoch in [args.epochs * 0.26, args.epochs * 0.4, args.epochs * 0.6, args.epochs * 0.83]:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        # if args.lr_scheduler == "default":
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.lr * (0.5 ** (epoch // args.lr_decay))
        # elif args.lr_scheduler == "cosine":
        #     scheduler.step()

        train(train_loader,criterion, optimizer, epoch)
        _, prec1 = test(model, criterion, test_loader)

        if prec1 > max(all_acc):
            print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
            torch.save(model.state_dict(), "./model/cifar10_{}{}_acc_{:.3f}_{}.pt".format(args.arch, args.depth, prec1, args.optmzr))
            print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(all_acc)))
            if len(all_acc) > 1:
                os.remove("./model/cifar10_{}{}_acc_{:.3f}_{}.pt".format(args.arch, args.depth, max(all_acc), args.optmzr))
        all_acc.append(prec1)

        if prec1 >= 99.99:
            print("accuracy is high ENOUGH!")
            break


    print("Best accuracy: " + str(max(all_acc)))


if __name__ == '__main__':
    main()