import torch      
import torch.nn as nn
from torchvision import datasets, models, transforms     
import torchvision.transforms as transforms              
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import cv2
import time
import sys
import os
import copy
import GPUtil
import shutil
import csv

from defectDataset import DefectDataset, ASUDepth
from network import *
# from network_rgbd import *
from visdom import Visdom
from matplotlib import pyplot as plt
import utils
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from shutil import copyfile
from writer import *


import argparse


# Optional arguments : Modifiables for shell script - training set size only for now
parser = argparse.ArgumentParser()
# parser.add_argument("--root_dir", 
# help = "Specify training data path Eg: '/home/rrathnak/Documents/Work/Task-2/Datasets/asu_cropped'")
# parser.add_argument("--dataset")
parser.add_argument("--num_training", help = 'Number of training samples', default=200)
args = parser.parse_args()

repeats = 3
num_epochs = 1500
root_dir = '/home/rrathnak/Documents/Work/Task-2/Datasets/asu_cropped'
dataset = 'ASU'
is_rgbd = True
val_dataset = 'ASU'
num_classes = 4
batch_size = 8
lr = 0.01
momentum = 0.9
optim_w_decay = 1e-5
step_size = 1500
gamma = 0.1
load_ckp = False
print_gpu_usage = False
dropout_prob = 0.1
mc_samples = 1
num_training = int(args.num_training)
optimizer_name = 'SGD'
directory_name = str(dataset) + '_PredictiveVariance_' + str(dropout_prob) + 'dropout_'+ str(num_training) + 'Train' + '_ASUColor' 
save_dir_name = 'Training_' + directory_name + '_' + 'Val_' + str(val_dataset)

for r in range(repeats):
    # Initialize plotter
    global plotter
    plotter = utils.VisdomLinePlotter(env_name=directory_name)

    # Create results directories
    savedir = directory_name + "_" + str(r)
    print("Savedir:", savedir)
    model_dir = os.path.join('models/', savedir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    best_dir = os.path.join(model_dir, 'best/')
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)


    # create config file
    copyfile('config.py', model_dir + '_config.py')


    # Activate GPU
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    # Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        }
    # Dataloaders

    image_datasets = {x: DefectDataset(root_dir, image_set = x, num_classes = num_classes, num_training = num_training, transforms=data_transforms[x])
                            for x in ['train', 'val']}
    dataloader = {x: DataLoader(image_datasets[x], batch_size= batch_size, shuffle=True, num_workers=0)
                        for x in ['train', 'val']}

    # Network
    vgg_model = VGGNet()
    net = FCNs(pretrained_net= vgg_model, n_class = num_classes, p = dropout_prob)
    vgg_model = vgg_model.to(device)
    net = net.to(device)


    # Optimizers
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=optim_w_decay,
                                    nesterov=False)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of gamma every step_size epochs
        # scheduler = lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = gamma)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = lr, weight_decay = optim_w_decay)


    if load_ckp:
        print("Loading checkpoint from previous save file...")
        ckp_path = checkpoint_dir + 'checkpoint.pt'
        net, optimizer, epoch = utils.load_ckp(ckp_path, net, optimizer=optimizer)
        print("Epoch loaded: ", epoch)
    # supervised loss
    sup_loss = nn.NLLLoss()
    global global_step
    global_step = 0
    best_IU = 0
    loss_history = []
    avg_loss = 0
    d_loss = 1e10
    break_out_flag = False
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs - 1))
        for phase in ['train','val']:
            batchIU = []
            batchF1 = []
            batch_accuracy = []
            running_acc = 0
            running_loss = 0
            print("Current phase: ", phase)
            if phase == 'train':
                net.train()
            else:
                net.eval()
                net.dropout.train()
            # Train/val loop
            # for iter, (input, depth, target, label) in enumerate(dataloader[phase]):
            for iter, (input, target, label) in enumerate(dataloader[phase]):
                input = input.to(device)
                # depth = depth.to(device)
                target = target.to(device)
                label = label.to(device)
                if phase == 'train':
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        # out = net(input,depth) # pass all inputs through encoder first
                        out = net(input) # pass all inputs through encoder first
                    y_predict = out[:, :num_classes, :, :]
                    var = out[:, num_classes:, :, :]

                    Lx = utils.stochastic_log_softmax(y_predict, var)
                    loss = sup_loss(Lx, target)
                    # Below, I have just replaced out with 'y_predict', the predicted output
                    out_ = y_predict.detach().clone()
                    label_ = label.detach().clone()
                    # sup_out__ = sup_out_.argmax(dim = 1).cpu()
                    # target__ = target_.argmax(dim = 1).cpu()
                    accuracy = utils.pixel_accuracy(out_, label_)
                    iou = utils.iou(out_, label_)
                    batchIU.append(iou)
                    batchF1.append(utils.f1(iou))
                    running_acc += np.mean(accuracy)
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    global_step += 1

                    print("Loss: ", loss.item(), "Batch_IU: ", iou, "Batch_F1:", utils.f1(iou))        

                else:
                    with torch.no_grad():
                        softmax = nn.Softmax(dim = 1)
                        outs = []
                        outs_sm = []
                        samples_IU = []
                        samples_F1 = []
                        samples_acc = []
                        for i in range(mc_samples):
                            # outs.append(net(input, depth))
                            outs.append(net(input))
                        # Metrics:
                        for out in outs:
                            y_predict = out[:, :num_classes, :, :]
                            var = out[:, num_classes:, :, :]
                            out_ = y_predict.detach().clone()
                            label_ = label.detach().clone()
                            out__ = softmax(out_)
                            outs_sm.append(out__.cpu().numpy())
                            accuracy = utils.pixel_accuracy(out_,label_) # batch accuracy
                            iou = utils.iou(out_, label_)
                            samples_acc.append(accuracy)
                            samples_IU.append(iou)
                            samples_F1.append(utils.f1(iou))
                        batch_accuracy.append(np.nanmean(np.asarray(samples_acc)))
                        batchIU.append(np.nanmean(np.asarray(samples_IU)))
                        batchF1.append(np.nanmean(np.asarray(samples_F1)))
                        Lx = utils.stochastic_log_softmax(y_predict, var)
                        loss = sup_loss(Lx, target)
                        running_acc += np.mean(batch_accuracy)
                        running_loss +=  loss.item()
            epoch_acc = running_acc/(iter+1)
            epoch_loss = running_loss/(iter + 1)
            epoch_IU = np.mean(batchIU)
            epoch_F1 = np.mean(batchF1)
            if optimizer_name == 'SGD' and phase == 'train':
                scheduler.step()
            if phase == 'val':
                loss_history.append(epoch_loss)
            print('{} Loss: {:.4f}, Acc: {:.4f}, IoU: {:.4f}, F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_IU, epoch_F1))
            
            plotter.plot('loss', phase, 'Loss', epoch, epoch_loss)
            plotter.plot('acc', phase, 'Accuracy', epoch, epoch_acc)
            plotter.plot('IU', phase, 'IU', epoch, epoch_IU)
            plotter.plot('F1', phase, 'F1', epoch, epoch_F1)

            if phase == 'val' and epoch_IU > best_IU:
                best_IU = epoch_IU
                is_best = True
            else:
                is_best = False
            checkpoint = {
                'epoch': epoch + 1,
                'net_state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            utils.save_ckp(checkpoint, is_best, checkpoint_dir, best_dir)       

            if len(loss_history) == 50:
                d_loss = np.abs(np.mean(np.stack(loss_history)) - avg_loss)
                avg_loss = np.mean(np.asarray(loss_history))
                loss_history = []

            if d_loss < 0.005:
                print('Average loss in the past 50 iterations:', avg_loss)
                print('d_loss: ', d_loss)
                break_out_flag = True
                utils.save_ckp(checkpoint, is_best, checkpoint_dir, best_dir)
                break
        if break_out_flag:
            print("Stopping early, no significant change in val loss.")
            break




