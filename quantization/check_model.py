import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from testers import *
from vgg import VGG
from collections import OrderedDict
import numpy as np
from mobilenetv2 import MobileNetV2


kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])),
    batch_size=256, shuffle=True, **kwargs)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def plot_heatmap(model):
    for name, weight in model.named_parameters():
        matrix1 = []
        weight = weight.cpu().detach().numpy()
        if len(weight.shape) == 4:
            # for row in range(weight.shape[0]):
            #     temp = []
            #     for column in range(weight.shape[1]):
            #         temp.append(0)
            #     matrix1.append(temp)
            # for row in range(weight.shape[0]):
            #     for column in range(weight.shape[1]):
            #         if np.sum(weight[row, column, :, :]) == 0:
            #             matrix1[row][column] = 0
            #         else:
            #             matrix1[row][column] = 1

            weight2d = weight.reshape(weight.shape[0], -1)
            im = plt.matshow(np.abs(weight2d), cmap=plt.cm.BuPu, aspect='auto')
            plt.colorbar(im)
            plt.title(name)
            # plt.savefig("filter1.png", dpi=800)
            plt.show()

def plot_distribution(model):
    font = {'size': 5}

    plt.rc('font', **font)

    fig = plt.figure(dpi=300)
    i = 1
    for name, weight in model.named_parameters():
        weight = weight.cpu().detach().numpy()
        if len(weight.shape) == 4:
            ax = fig.add_subplot(6, 10, i)
            weight = weight.reshape(1, -1)[0]
            xtick = np.linspace(-0.2, 0.2, 100)
            ax.hist(weight, bins=xtick)
            ax.set_title(name)
            i += 1
    plt.show()

def plot_distribution2(model):
    font = {'size': 5}

    plt.rc('font', **font)

    fig = plt.figure(dpi=300)
    i = 1
    for name, weight in model.named_parameters():
        weight = np.abs(weight.cpu().detach().numpy())
        weight2d = weight.reshape(weight.shape[0], -1)
        column_max = np.max(weight2d, axis=0)
        if len(weight.shape) == 4:
            xtict_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            xx = []
            yy = []
            for j, ticks in enumerate(xtict_values):
                if j == 0:
                    xx.append("<" + str(ticks))
                    yy.append(np.sum(column_max < ticks))
                if j != 0 and j != (len(xtict_values) - 1):
                    xx.append(str(xtict_values[j - 1]) + "~" + str(ticks))
                    yy.append(len(np.where(np.logical_and(column_max >= xtict_values[j - 1], column_max < ticks))[0]))
                if j == (len(xtict_values) - 1):
                    xx.append(">=" + str(ticks))
                    yy.append(np.sum(column_max >= ticks))
            ax = fig.add_subplot(3, 5, i)
            ax.bar(xx, yy, align='center', color="crimson")  # A bar chart
            ax.set_title(name)
            plt.setp(ax, xticks=xx)
            plt.xticks(rotation=90)
            i += 1
    plt.show()


def dataParallel_converter(model, model_path):
    # """
    #     convert between single gpu model and molti-gpu model
    # """
    # state_dict = torch.load(model_path)
    # new_state_dict = OrderedDict()

    # for k, v in state_dict.items():
    #     k = k.replace('module.', '')
    #     new_state_dict[k] = v

    # # model = torch.nn.DataParallel(model)
    # model.load_state_dict(new_state_dict)
    ################
    ## single GPU ##
    ################

    # model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(original_model_name).items()})
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] 
        # new_state_dict[name] = v
        # if 'module' not in k:
        #     k = 'module.'+ k
        # else:
        #     k = k.replace('features.module.', 'module.features.')
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)  # need basline model

    ################
    ## multi GPUs ##
    ################

    # cudnn.benchmark = True
    # model.load_state_dict(torch.load(original_model_name))
    model.cuda()

    # torch.save(model.state_dict(), './model/cifar10_vgg16_acc_93.540_3fc_sgd_in_multigpu.pt')

    return model

def main():

    model = MobileNetV2()
    #model = VGG(depth=16, init_weights=True, cfg=None)
    model = dataParallel_converter(model, "./model_reweighted/epochFinish.pt")
    #model = dataParallel_converter(model, "./model/cifar100_mobilenet_v2_exp3_acc_81.920_adam.pt")
    #model = dataParallel_converter(model, "./model_retrained/cifar10_vgg16_retrained_acc_91.310_config_vgg16_filter.pt")
    #model = dataParallel_converter(model, "./model_retrained/cifar100_mobilenetv217_retrained_acc_79.780_config_mobile_v2_0.7_threshold.pt")


    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):
            print(name, weight)
    print("\n------------------------------\n")

    test(model, device, test_loader)

    test_column_sparsity(model)
    # test_filter_sparsity(model)

    # plot_heatmap(model)
    plot_distribution(model)

    current_lr = 0.01
    rew_milestone = [100, 200, 300, 400]
    xx = np.linspace(1, 400, 400)
    yy = []
    for x in xx:
        if x - 1 in rew_milestone:
            current_lr *= 1.8
        else:
            current_lr *= 0.988
        yy.append(current_lr)
    plt.plot(xx, yy)
    # plt.show()

if __name__ == '__main__':
    main()

