import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet
import numpy as np

class Net(torch.nn.Module):

	def __init__(self, primaryEmbeddingSize, pretrained=True):
		super(Net, self).__init__()
		self.network = alexnet(pretrained=pretrained)
		self.network.classifier = nn.Sequential(*(self.network.classifier[i] for i in range(6)))
		self.network.classifier.add_module('6', nn.Linear(4096, primaryEmbeddingSize))
	
	def forward_once(self, x):
		"""
		Takes as input a 3*224*224 image, returns an embedding of length primaryEmbeddingSize.
		"""
		#x = F.normalize(self.network(x), p=2)
		x = self.network(x)
		return x

	def forward(self, image1, image2, image3):
		"""
		Returns pair of embeddings for pair of training images.
		"""
		output1 = self.forward_once(image1)
		output2 = self.forward_once(image2)
		output3 = self.forward_once(image3)
		return output1, output2, output3
