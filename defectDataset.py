import torch
import torchvision
import numpy as np
import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

import matplotlib
# matplotlib.use('tkAgg')
from matplotlib import pyplot as plt


class RoadCracks(Dataset):

	def __init__(self, root_dir, image_set='train', num_training = None, transforms=None):
		"""
		Parameters:
			root_dir (string): Root directory of the dumped NYU-Depth dataset.
			image_set (string, optional): Select the image_set to use, ``train``, ``val``
			transforms (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.n_class = 2
		self.root_dir = root_dir
		self.image_set = image_set
		self.transforms = transforms

		self.images = []
		self.depths = []
		self.targets = []
		if num_training and image_set == 'train':
			img_list = np.random.choice(self.read_image_list(os.path.join(root_dir, '{:s}.txt'.format(image_set))), num_training)
		else:
			img_list = self.read_image_list(os.path.join(root_dir, '{:s}.txt'.format(image_set)))
		for img_name in img_list:
			img_filename = os.path.join(root_dir, 'images/{:s}'.format(img_name))
			target_filename = os.path.join(root_dir, 'labels/{:s}'.format(img_name))

			if os.path.isfile(img_filename) and os.path.isfile(target_filename):
				self.images.append(img_filename)
				self.targets.append(target_filename)

	def read_image_list(self, filename):
		"""
		Read one of the image index lists

		Parameters:
			filename (string):  path to the image list file

		Returns:
			list (int):  list of strings that correspond to image names
		"""
		list_file = open(filename, 'r')
		img_list = []

		while True:
			next_line = list_file.readline()

			if not next_line:
				break

			img_list.append(next_line.rstrip())

		return img_list

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		image = Image.open(self.images[index]).convert('RGB')
		target = Image.open(self.targets[index])
		target = target.convert('L')
		
		if self.transforms is not None:
			image = self.transforms(image)
		
		resize = transforms.Resize((224,224), interpolation=Image.NEAREST)
		totensor = transforms.ToTensor()
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

		# Resize target, Convert target to numpy and one-hot encode
		target = resize.__call__(target)

		target = np.asarray(target)
		target = torch.from_numpy(target).long()
		h,w = target.size()
		
		label = torch.zeros(self.n_class, h, w)
		for c in range(self.n_class):
			label[c][target == c] = 1
		return image, target, label


class NEUDefects(Dataset):
	def __init__(self, root_dir, image_set='train', num_training = None, transforms=None):
		"""
		Parameters:
			root_dir (string): Root directory of the dumped NYU-Depth dataset.
			image_set (string, optional): Select the image_set to use, ``train``, ``val``
			transforms (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.n_class = 4
		self.root_dir = root_dir
		self.image_set = image_set
		self.transforms = transforms

		self.images = []
		self.depths = []
		self.targets = []
		if num_training and image_set == 'train':
			img_list = np.random.choice(self.read_image_list(os.path.join(root_dir, '{:s}.txt'.format(image_set))), num_training)
		else:
			img_list = self.read_image_list(os.path.join(root_dir, '{:s}.txt'.format(image_set)))
		for img_name in img_list:
			img_filename = os.path.join(root_dir, 'images/{:s}'.format(img_name))
			target_filename = os.path.join(root_dir, 'labels/{:s}'.format(img_name))

			if os.path.isfile(img_filename) and os.path.isfile(target_filename):
				self.images.append(img_filename)
				self.targets.append(target_filename)

	def read_image_list(self, filename):
		"""
		Read one of the image index lists

		Parameters:
			filename (string):  path to the image list file

		Returns:
			list (int):  list of strings that correspond to image names
		"""
		list_file = open(filename, 'r')
		img_list = []

		while True:
			next_line = list_file.readline()

			if not next_line:
				break

			img_list.append(next_line.rstrip())

		return img_list

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		image = Image.open(self.images[index]).convert('RGB')
		target = Image.open(self.targets[index])
		target = target.convert('L')
		
		if self.transforms is not None:
			image = self.transforms(image)
		
		resize = transforms.Resize((224,224), interpolation=Image.NEAREST)
		totensor = transforms.ToTensor()
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

		# Resize target, Convert target to numpy and one-hot encode
		target = resize.__call__(target)

		target = np.asarray(target)
		target = torch.from_numpy(target).long()
		h,w = target.size()
		
		label = torch.zeros(self.n_class, h, w)
		for c in range(self.n_class):
			label[c][target == c] = 1
		return image, target, label
