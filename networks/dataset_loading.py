from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop, Compose
from itertools import izip_longest
import numpy as np
from torch import cat

class SiameseNetworkDataset:

	def __init__(self, imageFile, inputSize, minibatchSize):
		self.inputSize = inputSize
		self.imageFile = imageFile
		self.minibatchSize = minibatchSize
		self.resize = Resize((self.inputSize, self.inputSize))
		self.totensor = ToTensor()
		# self.transforms = Compose([RandomCrop(self.inputSize), ToTensor(),
		self.transforms = Compose([ToTensor(),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		
	def __getitem__(self):
		with open(self.imageFile) as f:
			# for next_n_lines in izip_longest(*[f] * self.minibatchSize):

			self.compatibility_buffer = {x:[[],[]] for x in ['bt', 'tb', 'sb', 'bs', 'st', 'ts']}
			self.compatibility_labels = {x:[] for x in ['bt', 'tb', 'sb', 'bs', 'st', 'ts']}
			self.similarity_buffer = [[],[]]
			self.similarity_labels = []

			for line in f:
			
				# for line in next_n_lines:
				# im1, im2, label = line.rstrip().split()
				eg_info = line.rstrip().split()
				im1 = Image.open(eg_info[1])
				im2 = Image.open(eg_info[2])
	
				# if min(im1.size) < self.inputSize:
				im1 =  self.resize.__call__(im1)
				# if min(im2.size) < self.inputSize:
				im2 = self.resize.__call__(im2)
				
				objective = eg_info[0]
				if eg_info=='C':
					category_pair = eg_info[4]
					self.compatibility_buffer[category_pair][0].append(im1)
					self.compatibility_buffer[category_pair][1].append(im2)
					self.compatibility_labels[category_pair].append(eg_info[3])

					if len(self.compatibility_labels[category_pair])==self.minibatchSize:
						eg = self.yieldFunction(objective, category_pair)
						yield eg
						# im1Tensor = self.transforms.__call__(np.array(self.compatibility_buffer[category_pair][0]))
						# im2Tensor = self.transforms.__call__(np.array(self.compatibility_buffer[category_pair][1]))
						# labelTensor = self.totensor.__call__(np.array(self.compatibility_labels[category_pair]))
						# self.compatibility_buffer[category_pair][0] = []
						# self.compatibility_buffer[category_pair][1] = []
						# self.compatibility_labels[category_pair] = []
						# yield [objective, im1Tensor, im2Tensor, labelTensor, category_pair]

				else:
					self.similarity_buffer[0].append(im1)
					self.similarity_buffer[1].append(im2)
					self.similarity_labels.append(eg_info[3])

					if len(self.similarity_labels)==self.minibatchSize:
						eg = self.yieldFunction(objective)
						yield eg
						# im1Tensor = self.transforms.__call__(np.array(self.similarity_buffer[0]))
						# im2Tensor = self.transforms.__call__(np.array(self.similarity_buffer[1]))
						# labelTensor = self.totensor.__call__(np.array(self.similarity_labels))
						# self.similarity_buffer[0] = []
						# self.similarity_buffer[1] = []
						# self.similarity_labels = []
						# yield [objective, im1Tensor, im2Tensor, labelTensor]

				# im1 = self.transforms.__call__(im1)
				# im2 = self.transforms.__call__(im2)

				# yield im1, im2, int(label)   	
	 			# yield [eg_info[0]] + [im1, im2] + eg_info[3:]


	 		for category_pair in self.compatibility_buffer.keys():
	 			if len(self.compatibility_labels[category_pair])==0:
	 				continue
	 			eg = self.yieldFunction(objective, category_pair)
	 			yield eg
	 			# im1Tensor = self.transforms.__call__(np.array(self.compatibility_buffer[category_pair][0]))
				# im2Tensor = self.transforms.__call__(np.array(self.compatibility_buffer[category_pair][1]))
				# labelTensor = self.totensor.__call__(np.array(self.compatibility_labels[category_pair]))
				# yield ['C', im1Tensor, im2Tensor, labelTensor, category_pair]

			if len(self.similarity_labels)>0:
				eg = self.yieldFunction(objective)
				yield eg
				# im1Tensor = self.transforms.__call__(np.array(self.similarity_buffer[0]))
				# im2Tensor = self.transforms.__call__(np.array(self.similarity_buffer[1]))
				# labelTensor = self.totensor.__call__(np.array(self.similarity_labels))
				# yield ['S', im1Tensor, im2Tensor, labelTensor] 


	def yieldFunction(self, objective, category_pair=None):

		if objective=='C'
			im1Tensor = cat((self.transforms.__call__(x) for x in self.compatibility_buffer[category_pair][0]), 0)
			im2Tensor = cat((self.transforms.__call__(x) for x in self.compatibility_buffer[category_pair][1]), 0)
			labelTensor = self.totensor.__call__(np.array(self.compatibility_labels[category_pair]))
			self.compatibility_buffer[category_pair][0]=[]
			self.compatibility_buffer[category_pair][1]=[]
			self.compatibility_labels[category_pair] = []
			return [objective, im1Tensor, im2Tensor, labelTensor, category_pair]
		else:
			im1Tensor = cat((self.transforms.__call__(x) for x in self.similarity_buffer[0]), 0)
			im2Tensor = cat((self.transforms.__call__(x) for x in self.similarity_buffer[1]), 0)
			labelTensor = self.totensor.__call__(np.array(self.similarity_labels))
			self.similarity_buffer[0] = []
			self.similarity_buffer[1] = []
			self.similarity_labels = []
			return ['S', im1Tensor, im2Tensor, labelTensor] 
