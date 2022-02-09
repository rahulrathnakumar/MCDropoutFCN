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
from defectDataset import DefectDataset, ASUDepth
from network import *
from network_rgbd import *
from visdom import Visdom
from matplotlib import pyplot as plt
from utils import *
from PIL import Image
from matplotlib import cm

import time 
import utils



import argparse


# Optional arguments : Modifiables for shell script - training set size only for now
parser = argparse.ArgumentParser()
# parser.add_argument("--root_dir", 
# help = "Specify training data path Eg: '/home/rrathnak/Documents/Work/Task-2/Datasets/asu_cropped'")
# parser.add_argument("--dataset")
parser.add_argument("--num_training", help = 'Number of training samples')
args = parser.parse_args()


num_epochs = 1500
root_dir = '/home/rrathnak/Documents/Work/Task-2/Datasets/concreteCrack'
dataset = 'RoadCracks'
is_rgbd = False
val_dataset = 'ConcreteCrack'
num_classes = 2
batch_size = 8
lr = 0.01
momentum = 0.9
optim_w_decay = 1e-5
step_size = 1500
gamma = 0.1
print_gpu_usage = False
dropout_prob = 0.1
mc_samples = 80
num_training = 120
repeats = 3
optimizer_name = 'SGD'
directory_name = str(dataset) + '_' + str(dropout_prob) + 'dropout_'+ str(num_training) + 'Train' + '_StepLR'
save_dir_name = 'Training_' + directory_name + '_' + 'Val_' + str(val_dataset)




print(args.num_training)

repeats = repeats

epistemic_uncertainty_full = []
epistemic_F1_full = []
preds_full = []
for r in range(repeats):
	epistemic_uncertainty_repeat = []
	preds_repeat = []
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

	loaddir = directory_name + "_" + str(r)
	save_dir_name = save_dir_name
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
	if is_rgbd:
		val_dataset = ASUDepth(root_dir = root_dir, num_classes = num_classes, image_set='val', transforms= data_transforms['val'])
	else:
		val_dataset = DefectDataset(root_dir = root_dir, num_classes = num_classes, image_set='val', transforms= data_transforms['val'])
	val_dataloader = DataLoader(val_dataset, batch_size= batch_size, shuffle=False)

	vgg_model = VGGNet()
	if is_rgbd:
		net = FCNDepth(pretrained_net=vgg_model, n_class = num_classes, p = p)
	else:
		net = FCNs(pretrained_net=vgg_model, n_class = num_classes, p = p)
	best_path = best_dir + 'best_model.pt'
	net, epoch = utils.load_ckp(best_path, net)
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
		# for iter, (input, depth, target, label) in enumerate(val_dataloader):
		for iter, (input, target, label) in enumerate(val_dataloader):
			outs = []
			outs_sm = []
			samples_IU = []
			samples_F1 = []
			samples_acc = []

			# Epistemic Uncertainty
			input = input.to(device)
			# depth = depth.to(device)
			target = target.to(device)
			label = label.to(device)
			# Store image and label here...
			if r == 0:
				img = input.detach().cpu()
				img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)			
				img = img.numpy().transpose(0,2,3,1)


			for i in range(num_samples):
				# outs.append(net(input, depth))        
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
			
			mean_output = torch.mean(torch.stack(outs), dim = 0)
			N, _, h, w = mean_output.shape
			mean_output = mean_output.detach().cpu().numpy()
			pred = mean_output.transpose(0, 2, 3, 1).reshape(-1, num_classes).argmax(axis=1).reshape(N, h, w)
			pred = normalize(pred)
			preds_repeat.append(pred)
			y_sm = np.nanmean(np.asarray(outs_sm), axis = 0)
			epistemic_uncertainty = utils.entropy(y_sm)
			batch_accuracy.append(np.nanmean(np.asarray(samples_acc), axis = 0))
			batchIU.append(np.nanmean(np.asarray(samples_IU), axis = 0))
			batchF1.append(np.nanmean(np.asarray(samples_F1), axis = 0))
			epistemic_uncertainty_repeat.append(epistemic_uncertainty)


	preds_full.append(np.vstack(preds_repeat))
	epistemic_uncertainty_full.append(np.vstack(epistemic_uncertainty_repeat))
	epistemic_F1_full.append(np.hstack(batchF1))
preds_averaged = np.nanmean(np.asarray(preds_full), axis = 0)
epistemic_uncertainty_averaged = np.nanmean(np.asarray(epistemic_uncertainty_full), axis = 0)
epistemic_F1_averaged = np.mean(np.asarray(epistemic_F1_full), axis = 0)


# Compute average epistemic uncertainty per image
mean_epi_uncertainty = []
for epi in epistemic_uncertainty_averaged:
	mean_epi_uncertainty.append(np.mean(epi))
print("Mean epistemic uncertainty: ", np.mean(mean_epi_uncertainty))
print("Mean F1 score (epistemic): ", np.mean(epistemic_F1_averaged))
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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + (torch.randn(tensor.size()) * self.std + self.mean).cuda()
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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
	is_rgbd = is_rgbd
	num_classes = num_classes
	# MC Sampling
	num_samples = mc_samples
	# Admin
	load_model = True

	assert load_model, "Cannot load model. (load_model = False)"

	loaddir = directory_name + "_" + str(r)
	save_dir_name = save_dir_name
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
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
	}
	# Dataloading
	if is_rgbd:
		val_dataset = ASUDepth(root_dir = root_dir, num_classes = num_classes, image_set='val', transforms= data_transforms['val'])
	else:
		val_dataset = DefectDataset(root_dir = root_dir, num_classes = num_classes, image_set='val', transforms= data_transforms['val'])
	val_dataloader = DataLoader(val_dataset, batch_size= batch_size, shuffle=False)

	vgg_model = VGGNet()
	if is_rgbd:
		net = FCNDepth(pretrained_net=vgg_model, n_class = num_classes, p = p)
	else:
		net = FCNs(pretrained_net=vgg_model, n_class = num_classes, p = p)
	tta_model = tta.SegmentationTTAWrapper(net, tta.aliases.d4_transform(), merge_mode='mean')
	best_path = best_dir + 'best_model.pt'
	net, epoch = utils.load_ckp(best_path, net)
	print("Epoch loaded: ", epoch)
	vgg_model = vgg_model.to(device)
	net = net.to(device)
	net.eval()
	mean=[0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	print("Validating at epoch: {:.4f}".format(epoch))
	noise = AddGaussianNoise(mean = 0, std = 0.5)
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
				# Add random noise ...
				input = noise(input)
				# outs.append(net(input, depth))
				# outs_unaug.append(net(input, depth))
				outs.append(net(input))
				# outs_unaug.append(net(input))

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

			# for out in outs_unaug:
			# 	N, _, h, w = out.shape
			# 	out_ = out.detach().clone()
			# 	label_ = label.detach().clone()
			# 	out__ = softmax(out_)
			# 	outs_sm_unaug.append(out__.cpu().numpy())
			# 	accuracy = utils.pixel_accuracy(out_,label_) # batch accuracy
			# 	iou = utils.iou(out_, label_, per_image=True)
			# 	samples_acc_unaug.append(accuracy)
			# 	samples_IU_unaug.append(iou)
			# 	samples_F1_unaug.append(utils.f1(iou))

			y_sm = np.nanmean(np.asarray(outs_sm), axis = 0)
			# y_sm_unaug = np.nanmean(np.asarray(outs_sm_unaug), axis = 0)
			aleatoric_uncertainty = np.abs(utils.entropy(y_sm))
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
print("Mean aleatoric uncertainty: ", np.mean(mean_ale_uncertainty))
print("Mean F1 score (aleatoric): ", np.mean(aleatoric_F1_averaged))


# Plot results for aleatoric
count = 0
for ale in aleatoric_uncertainty_averaged:
		count += 1
		save_predictions(imgList = [ale], path = save_dir + '/' + str(count) + "_aleAveraged.png")

# Plot scatter F1 vs Mean aleatoric uncertainty
plt.scatter(aleatoric_F1_averaged, mean_ale_uncertainty)
plt.xlabel('F1-score averaged from Test Time Augmentation')
plt.ylabel('Average Aleatoric Uncertainty')
plt.savefig(save_dir + '/' + 'ale_F1.png')
plt.close()

# Open file for writing in append mode
with open(str(dataset) + '_' + str(dropout_prob) + 'dropout_' + '80mcsamples_concreteCrackTesting_uq_112821.txt' , 'a') as file:
	file.write('Model with ' + str(args.num_training) + ' examples: \n')
	file.write("Mean aleatoric uncertainty: " + str(np.mean(mean_ale_uncertainty)) + "\n")
	file.write("Mean F1 score (Aleatoric) :" + str(np.mean(aleatoric_F1_averaged)) + "\n") 
	file.write("Mean epistemic uncertainty: " + str(np.mean(mean_epi_uncertainty)) + "\n")
	file.write("Mean F1 score (epistemic) :" + str(np.mean(epistemic_F1_averaged)) + "\n") 
	file.write("---------------------------------------------------------------- \n")
	file.close()

