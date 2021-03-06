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
from config import *
from defectDataset import ASUDepth, DefectDataset
from network_rgbd import *
from network import *
from visdom import Visdom
from matplotlib import pyplot as plt
from utils import *
from PIL import Image
from matplotlib import cm

import time 
import utils



epistemic_uncertainty_full = []
batchIU = []
batchF1 = []
batch_accuracy = []


# CONFIG VARIABLES
p = dropout_prob
# Dataset parameters
root_dir = root_dir
num_classes = num_classes
is_rgbd = is_rgbd
# MC Sampling
num_samples = mc_samples

# Admin
load_model = True


# Initialize plotter
global plotter
plotter = utils.VisdomLinePlotter(env_name='main')

assert load_model, "Cannot load model. (load_model = False)"

loaddir = directory_name
save_dir_name = directory_name
print("Current Directory:", loaddir)
model_dir = os.path.join('models/', loaddir)
checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
best_dir = os.path.join(model_dir, 'best/')
save_dir = 'results/' + save_dir_name
if not os.path.exists(save_dir):
	os.makedirs(save_dir)


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
# Dataloading
val_dataset = DefectDataset(root_dir = root_dir, num_classes = num_classes, image_set='val', transforms = data_transforms['val'])
val_dataloader = DataLoader(val_dataset, batch_size= batch_size)

vgg_model = VGGNet()
if is_rgbd:
	net = FCNDepth(pretrained_net=vgg_model, n_class = num_classes, p = p)
else:
	net = FCNs(pretrained_net=vgg_model, n_class = num_classes, p = p)
best_path = best_dir + 'best_model.pt'
net, epoch = load_ckp(best_path, net)
print("Epoch loaded: ", epoch)
vgg_model = vgg_model.to(device)
net = net.to(device)
net.eval()
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
print("Validating at epoch: {:.4f}".format(epoch))
with torch.no_grad():
	# Epistemic Uncertainty
	net.dropout.train()
	softmax = nn.Softmax(dim = 1)
	count = 0
	batchIU = []
	batchF1 = []
	batch_accuracy = []
	count = 0
	# for iter, (input, depth, target, label) in enumerate(val_dataloader):
	for iter, (input, target, label) in enumerate(val_dataloader):
		outs = []
		outs_sm = []
		samples_IU = []
		samples_F1 = []
		samples_acc = []

		input = input.to(device)
		# depth = depth.to(device)
		target = target.to(device)
		label = label.to(device)
		for i in range(num_samples):
			# outs.append(net(input, depth))        
			outs.append(net(input))
		for out in outs:
			out_ = out.detach().clone()
			label_ = label.detach().clone()
			out__ = softmax(out_)
			outs_sm.append(out__.cpu().numpy())
			accuracy = utils.pixel_accuracy(out_,label_) # batch accuracy
			iou = utils.iou(out_, label_, per_image=True)
			if np.isnan(iou).any():
				print(iou)
			samples_acc.append(accuracy)
			samples_IU.append(iou)
			samples_F1.append(utils.f1(iou))
	
		y_sm = np.nanmean(np.asarray(outs_sm), axis = 0)
		epistemic_uncertainty = utils.entropy(y_sm)
		batch_accuracy.append(np.nanmean(np.asarray(samples_acc), axis = 0))
		batchIU.append(np.nanmean(np.asarray(samples_IU), axis = 0))
		batchF1.append(np.nanmean(np.asarray(samples_F1), axis = 0))
		epistemic_uncertainty_full.append(epistemic_uncertainty)
		
		# Save predictions
		img = input.detach().cpu()
		img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)			
		img = img.numpy().transpose(0,2,3,1)
		mean_output = torch.mean(torch.stack(outs), dim = 0)
		N, _, h, w = mean_output.shape
		mean_output = mean_output.detach().cpu().numpy()
		pred = mean_output.transpose(0, 2, 3, 1).reshape(-1, num_classes).argmax(axis=1).reshape(N, h, w)
		pred = normalize(pred)
		gt = label.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, num_classes).argmax(axis=1).reshape(N, h, w)
		for i in range(N):
			count+=1
			save_predictions(imgList = [img[i,:,:,0], gt[i], pred[i], epistemic_uncertainty[i]], path = save_dir + '/' + str(count) + "_epiTiled.png")


batchF1 = np.hstack(batchF1)
epistemic_uncertainty_full = np.vstack(epistemic_uncertainty_full)

mean_epi_uncertainty = []
for epi in epistemic_uncertainty_full:
	mean_epi_uncertainty.append(np.mean(epi))


# Plot results for epistemic
count = 0
for epi in epistemic_uncertainty_full:
		count += 1
		save_predictions(imgList = [epi], path = save_dir + '/' + str(count) + "_epiAveraged.png")
print("Mean epistemic uncertainty: ", np.mean(mean_epi_uncertainty))
print("Mean F1 score (epistemic): ", np.mean(batchF1))

# Plot scatter F1 vs Mean epistemic uncertainty
plt.scatter(batchF1, mean_epi_uncertainty)
plt.xlabel('F1-score averaged from Test Time Augmentation')
plt.ylabel('Average Epistemic Uncertainty')
plt.savefig(save_dir + '/' + 'epi_F1.png')
plt.close()


import ttach as tta


# Aleatoric Uncertainty
aleatoric_uncertainty_full = []
batchIU = []
batchF1 = []
batch_accuracy = []


# CONFIG VARIABLES
p = dropout_prob
# Dataset parameters
root_dir = root_dir
num_classes = num_classes
is_rgbd = is_rgbd
# MC Sampling
num_samples = mc_samples
# Admin
load_model = True

assert load_model, "Cannot load model. (load_model = False)"

loaddir = directory_name
save_dir_name = directory_name
print("Current Directory:", loaddir)
model_dir = os.path.join('models/', loaddir)
checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
best_dir = os.path.join(model_dir, 'best/')
save_dir = 'results/' + save_dir_name
if not os.path.exists(save_dir):
	os.makedirs(save_dir)


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
# Dataloading
val_dataset = DefectDataset(root_dir = root_dir, num_classes = num_classes, image_set='val', transforms= data_transforms['val'])
val_dataloader = DataLoader(val_dataset, batch_size= batch_size, shuffle=False)

vgg_model = VGGNet()
if is_rgbd:
	net = FCNDepth(pretrained_net=vgg_model, n_class = num_classes, p = p)
else:
	net = FCNs(pretrained_net=vgg_model, n_class = num_classes, p = p)

tta_model = tta.SegmentationTTAWrapper(net, tta.aliases.d4_transform(), merge_mode='mean')
best_path = best_dir + 'best_model.pt'
net, epoch = load_ckp(best_path, net)
print("Epoch loaded: ", epoch)
vgg_model = vgg_model.to(device)
net = net.to(device)
net.eval()
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
print("Validating at epoch: {:.4f}".format(epoch))
with torch.no_grad():
	# for iter, (input, depth, target, label) in enumerate(val_dataloader):
	for iter, (input, target, label) in enumerate(val_dataloader):
		# Transform data using custom transforms function - mc_samples
		softmax = nn.Softmax(dim = 1)
		input = input.to(device)
		# depth = depth.to(device)
		target = target.to(device)
		label = label.to(device)
		outs = []
		outs_unaug = []
		outs_sm = []
		outs_sm_unaug = []
		samples_IU = []
		samples_IU_unaug = []
		samples_F1 = []
		samples_F1_unaug = []
		samples_acc = []
		samples_acc_unaug = []
		for i in range(num_samples):
			# outs.append(tta_model(input, depth))
			# outs_unaug.append(net(input, depth))
			outs.append(tta_model(input))
			outs_unaug.append(net(input))

		for out in outs:
			N, _, h, w = out.shape
			out_ = out.detach().clone()
			label_ = label.detach().clone()
			out__ = softmax(out_)
			outs_sm.append(out__.cpu().numpy())
			accuracy = utils.pixel_accuracy(out_,label_) # batch accuracy
			iou = utils.iou(out_, label_, per_image=True)
			samples_acc.append(accuracy)
			samples_IU.append(iou)
			samples_F1.append(utils.f1(iou))

		for out in outs_unaug:
			N, _, h, w = out.shape
			out_ = out.detach().clone()
			label_ = label.detach().clone()
			out__ = softmax(out_)
			outs_sm_unaug.append(out__.cpu().numpy())
			accuracy = utils.pixel_accuracy(out_,label_) # batch accuracy
			iou = utils.iou(out_, label_, per_image=True)
			samples_acc_unaug.append(accuracy)
			samples_IU_unaug.append(iou)
			samples_F1_unaug.append(utils.f1(iou))


		y_sm = np.nanmean(np.asarray(outs_sm), axis = 0)
		y_sm_unaug = np.nanmean(np.asarray(outs_sm_unaug), axis = 0)
		aleatoric_uncertainty = np.abs(utils.entropy(y_sm) - utils.entropy(y_sm_unaug))
		batch_accuracy.append(np.nanmean(np.asarray(samples_acc), axis = 0))
		batchIU.append(np.nanmean(np.asarray(samples_IU), axis = 0))
		batchF1.append(np.nanmean(np.asarray(samples_F1), axis = 0))
		aleatoric_uncertainty_full.append(aleatoric_uncertainty)


batchF1 = np.hstack(batchF1)
aleatoric_uncertainty_full = np.vstack(aleatoric_uncertainty_full)

# Compute average aleatoric uncertainty per image
mean_ale_uncertainty = []
for ale in aleatoric_uncertainty_full:
	mean_ale_uncertainty.append(np.mean(ale))


# Plot results for epistemic
count = 0
for ale in aleatoric_uncertainty_full:
		count += 1
		save_predictions(imgList = [ale], path = save_dir + '/' + str(count) + "_aleAveraged.png")

# Plot scatter F1 vs Mean epistemic uncertainty
plt.scatter(batchF1, mean_ale_uncertainty)
plt.xlabel('F1-score averaged from Test Time Augmentation')
plt.ylabel('Average Aleatoric Uncertainty')
plt.savefig(save_dir + '/' + 'ale_F1.png')
plt.close()

print("Mean aleatoric uncertainty: ", np.mean(mean_ale_uncertainty))
print("Mean F1 score (Ale): ", np.mean(batchF1))
