from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop, Compose
from itertools import izip_longest
import numpy as np

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

			compatibility_buffer = {x:[[],[]] for x in ['bt', 'tb', 'sb', 'bs', 'st', 'ts']}
			compatibility_labels = {x:[] for x in ['bt', 'tb', 'sb', 'bs', 'st', 'ts']}
			similarity_buffer = [[],[]]
			similarity_labels = []

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
					compatibility_buffer[category_pair][0].append(np.array(im1))
					compatibility_buffer[category_pair][1].append(np.array(im2))
					compatibility_labels[category_pair].append(eg_info[3])

					if len(compatibility_labels[category_pair])==self.minibatchSize:
						im1Tensor = self.transforms.__call__(np.array(compatibility_buffer[category_pair][0]))
						im2Tensor = self.transforms.__call__(np.array(compatibility_buffer[category_pair][1]))
						labelTensor = self.totensor.__call__(np.array(compatibility_labels[category_pair]))
						compatibility_buffer[category_pair][0] = []
						compatibility_buffer[category_pair][1] = []
						compatibility_labels[category_pair] = []
						yield [objective, im1Tensor, im2Tensor, labelTensor, category_pair]

				else:
					similarity_buffer[0].append(np.array(im1))
					similarity_buffer[1].append(np.array(im2))
					similarity_labels.append(eg_info[3])

					if len(similarity_labels)==self.minibatchSize:
						im1Tensor = self.transforms.__call__(np.array(similarity_buffer[0]))
						im2Tensor = self.transforms.__call__(np.array(similarity_buffer[1]))
						labelTensor = self.totensor.__call__(np.array(similarity_labels))
						similarity_buffer[0] = []
						similarity_buffer[1] = []
						similarity_labels = []
						yield [objective, im1Tensor, im2Tensor, labelTensor]

				# im1 = self.transforms.__call__(im1)
				# im2 = self.transforms.__call__(im2)

				# yield im1, im2, int(label)   	
	 			# yield [eg_info[0]] + [im1, im2] + eg_info[3:]


	 		for category_pair in compatibility_buffer.keys():
	 			if len(compatibility_labels[category_pair])==0:
	 				continue
	 			im1Tensor = self.transforms.__call__(np.array(compatibility_buffer[category_pair][0]))
				im2Tensor = self.transforms.__call__(np.array(compatibility_buffer[category_pair][1]))
				labelTensor = self.totensor.__call__(np.array(compatibility_labels[category_pair]))
				yield ['C', im1Tensor, im2Tensor, labelTensor, category_pair]

			if len(similarity_labels)>0:
				im1Tensor = self.transforms.__call__(np.array(similarity_buffer[0]))
				im2Tensor = self.transforms.__call__(np.array(similarity_buffer[1]))
				labelTensor = self.totensor.__call__(np.array(similarity_labels))
				yield ['S', im1Tensor, im2Tensor, labelTensor] 

