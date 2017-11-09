from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop, Compose
import numpy as np
from torch import stack, from_numpy

class SiameseNetworkDataset:

	def __init__(self, imageFile, inputSize, minibatchSize):
		self.inputSize = inputSize
		self.imageFile = imageFile
		self.minibatchSize = minibatchSize
		self.resize = Resize((self.inputSize, self.inputSize))
		self.totensor = ToTensor()
		self.transforms = Compose([ToTensor(),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		
	def __getitem__(self):
		with open(self.imageFile) as f:
			self.compatibility_buffer = {x:[[],[]] for x in ['bt', 'tb', 'sb', 'bs', 'st', 'ts']}
			self.compatibility_labels = {x:[] for x in ['bt', 'tb', 'sb', 'bs', 'st', 'ts']}
			self.similarity_buffer = [[],[]]
			self.similarity_labels = []

			for line in f:
				eg_info = line.rstrip().split()
				try:
					im1 = Image.open(eg_info[1])
				except IOError:
					continue
				try:
					im2 = Image.open(eg_info[2])
				except:
					continue
				im1 =  self.resize.__call__(im1)
				im2 = self.resize.__call__(im2)
				
				objective = eg_info[0]
				if objective=='C':
					category_pair = eg_info[4]
					self.compatibility_buffer[category_pair][0].append(im1)
					self.compatibility_buffer[category_pair][1].append(im2)
					self.compatibility_labels[category_pair].append(int(eg_info[3]))
					
					if len(self.compatibility_labels[category_pair])==self.minibatchSize:
						eg = self.yieldFunction(objective, category_pair=category_pair)
						yield eg

				else:
					self.similarity_buffer[0].append(im1)
					self.similarity_buffer[1].append(im2)
					self.similarity_labels.append(int(eg_info[3]))

					if len(self.similarity_labels)==self.minibatchSize:
						eg = self.yieldFunction(objective)
						yield eg

	 		for category_pair in self.compatibility_buffer.keys():
	 			if len(self.compatibility_labels[category_pair])==0:
	 				continue
	 			eg = self.yieldFunction('C', category_pair=category_pair)
	 			yield eg

			if len(self.similarity_labels)>0:
				eg = self.yieldFunction('S')
				yield eg

	def yieldFunction(self, objective, category_pair=None):
		if objective=='C':
			im1Tensor = stack([self.transforms.__call__(x) for x in self.compatibility_buffer[category_pair][0]], 0)
			im2Tensor = stack([self.transforms.__call__(x) for x in self.compatibility_buffer[category_pair][1]], 0)
			labelTensor = from_numpy(np.array(self.compatibility_labels[category_pair])).float()
			self.compatibility_buffer[category_pair][0]=[]
			self.compatibility_buffer[category_pair][1]=[]
			self.compatibility_labels[category_pair] = []
			return [objective, im1Tensor, im2Tensor, labelTensor, category_pair]
		else:
			im1Tensor = stack([self.transforms.__call__(x) for x in self.similarity_buffer[0]], 0)
			im2Tensor = stack([self.transforms.__call__(x) for x in self.similarity_buffer[1]], 0)
			labelTensor = from_numpy(np.array(self.similarity_labels)).float()
			self.similarity_buffer[0] = []
			self.similarity_buffer[1] = []
			self.similarity_labels = []
			return ['S', im1Tensor, im2Tensor, labelTensor]
