from numpy.core.fromnumeric import repeat
import torch      
import torch.nn as nn
from torchvision import datasets, models, transforms     
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
from defectDataset import DefectDataset
from network import *
from visdom import Visdom
from matplotlib import pyplot as plt
from utils import *
from PIL import Image
from matplotlib import cm

import time 
import utils


repeats = repeats

epistemic_uncertainty_full = []
epistemic_F1_full = []
for r in range(repeats):
	epistemic_uncertainty_repeat = []
	batchIU = []
	batchF1 = []
	batch_accuracy = []

	# CONFIG VARIABLES
	p = dropout_prob

	# Dataset parameters
	root_dir = root_dir
	num_classes = num_classes
	# MC Sampling
	num_samples = mc_samples
	# Admin
	load_model = True

	assert load_model, "Cannot load model. (load_model = False)"

	loaddir = directory_name + "_" + str(r)
	print("Current Directory:", loaddir)
	model_dir = os.path.join('models/', loaddir)
	checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
	best_dir = os.path.join(model_dir, 'best/')
	save_dir = 'results/' + directory_name
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

		for iter, (input, target, label) in enumerate(val_dataloader):
			outs = []
			outs_sm = []
			samples_IU = []
			samples_F1 = []
			samples_acc = []

			# Epistemic Uncertainty
			input = input.to(device)
			target = target.to(device)
			label = label.to(device)
			for i in range(num_samples):
				outs.append(net(input))        
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

			y_sm = np.nanmean(np.asarray(outs_sm), axis = 0)
			epistemic_uncertainty = utils.entropy(y_sm)
			batch_accuracy.append(np.nanmean(np.asarray(samples_acc), axis = 0))
			batchIU.append(np.nanmean(np.asarray(samples_IU), axis = 0))
			batchF1.append(np.nanmean(np.asarray(samples_F1), axis = 0))
			epistemic_uncertainty_repeat.append(epistemic_uncertainty)
	epistemic_uncertainty_full.append(np.vstack(epistemic_uncertainty_repeat))
	epistemic_F1_full.append(np.hstack(batchF1))
epistemic_uncertainty_averaged = np.nanmean(np.asarray(epistemic_uncertainty_full), axis = 0)
epistemic_F1_averaged = np.mean(np.asarray(epistemic_F1_full), axis = 0)


# Compute average epistemic uncertainty per image
mean_epi_uncertainty = []
for epi in epistemic_uncertainty_averaged:
	mean_epi_uncertainty.append(np.mean(epi))

# Plot results for epistemic
count = 0
for epi in epistemic_uncertainty_averaged:
		count += 1
		save_predictions(imgList = [epi], path = save_dir + '/' + str(count) + "_epiAveraged.png")

# Plot scatter F1 vs Mean epistemic uncertainty
plt.scatter(epistemic_F1_averaged, mean_epi_uncertainty)
plt.xlabel('F1-score averaged from MC Dropout')
plt.ylabel('Average Epistemic Uncertainty')
plt.savefig(save_dir + '/' + 'epi_F1.png')
plt.close()


import ttach as tta


# Aleatoric Uncertainty
aleatoric_uncertainty_full = []
aleatoric_F1_full = []

for r in range(repeats):
	aleatoric_uncertainty_repeat = []
	batchIU = []
	batchF1 = []
	batch_accuracy = []


	# CONFIG VARIABLES
	p = dropout_prob
	# Dataset parameters
	root_dir = root_dir
	num_classes = num_classes
	# MC Sampling
	num_samples = mc_samples
	# Admin
	load_model = True

	assert load_model, "Cannot load model. (load_model = False)"

	loaddir = directory_name + "_" + str(r)
	print("Current Directory:", loaddir)
	model_dir = os.path.join('models/', loaddir)
	checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
	best_dir = os.path.join(model_dir, 'best/')
	save_dir = 'results/' + directory_name
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
		for iter, (input, target, label) in enumerate(val_dataloader):
			# Transform data using custom transforms function - mc_samples
			softmax = nn.Softmax(dim = 1)
			input = input.to(device)
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
			aleatoric_uncertainty_repeat.append(aleatoric_uncertainty)
			batch_accuracy.append(np.nanmean(np.asarray(samples_acc), axis = 0))
			batchIU.append(np.nanmean(np.asarray(samples_IU), axis = 0))
			batchF1.append(np.nanmean(np.asarray(samples_F1), axis = 0))
	aleatoric_F1_full.append(np.hstack(batchF1))
	aleatoric_uncertainty_full.append(np.vstack(aleatoric_uncertainty_repeat))
aleatoric_uncertainty_averaged = np.nanmean(np.asarray(aleatoric_uncertainty_full), axis = 0)
aleatoric_F1_averaged = np.mean(np.asarray(aleatoric_F1_full), axis = 0)


# Compute average aleatoric uncertainty per image
mean_ale_uncertainty = []
for ale in aleatoric_uncertainty_averaged:
	mean_ale_uncertainty.append(np.mean(ale))


# Plot results for epistemic
count = 0
for ale in aleatoric_uncertainty_averaged:
		count += 1
		save_predictions(imgList = [ale], path = save_dir + '/' + str(count) + "_aleAveraged.png")

# Plot scatter F1 vs Mean epistemic uncertainty
plt.scatter(aleatoric_F1_averaged, mean_ale_uncertainty)
plt.xlabel('F1-score averaged from Test Time Augmentation')
plt.ylabel('Average Aleatoric Uncertainty')
plt.savefig(save_dir + '/' + 'ale_F1.png')
plt.close()