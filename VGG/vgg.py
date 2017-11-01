import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class Net(torch.nn.Module):

	def __init__(self, primaryEmbeddingSize, secEmbeddingSize):
		super(Net, self).__init__()
		self.vgg16 = vgg16(pretrained=True)
		self.vgg16.classifier = nn.Sequential(*(self.vgg16.classifier[i] for i in range(6)))
		self.vgg16.classifier.add_module('6', nn.Linear(4096, primaryEmbeddingSize))

		self.top_bottom = nn.Linear(primaryEmbeddingSize, secEmbeddingSize)
		self.top_shoe = nn.Linear(primaryEmbeddingSize, secEmbeddingSize)
		self.bottom_shoe = nn.Linear(primaryEmbeddingSize, secEmbeddingSize)

	def similarity_forward_once(self, x):
		"""
		Takes as input a 3*224*224 image, returns an embedding of length 4096.
		"""
		x = self.vgg16(x)
		return x

	def similarity_forward(self, image1, image2):
		"""
		Returns pair of embeddings for pair of training images.
		"""
		output1 = self.similarity_forward_once(image1)
		output2 = self.similarity_forward_once(image2)
		return output1, output2

	def top_bottom_forward_once(self, x):
		x = self.top_bottom(self.vgg16(x))
		return x

	def top_bottom_forward(self, image1, image2):
		output1 = self.top_bottom_forward_once(image1)
		output2 = self.top_bottom_forward_once(image2)
		return output1, output2

	def top_shoe_forward_once(self, x):
		x = self.top_shoe(self.vgg16(x))
		return x

	def top_shoe_forward(self, image1, image2):
		output1 = self.top_shoe_forward_once(image1)
		output2 = self.top_shoe_forward_once(image2)
		return output1, output2

	def bottom_shoe_forward_once(self, x):
		x = self.bottom_shoe(self.vgg16(x))
		return x

	def bottom_shoe_forward(self, image1, image2):
		output1 = self.bottom_shoe_forward_once(image1)
		output2 = self.bottom_shoe_forward_once(image2)
		return output1, output2

	def compatibility_forward(self, image1, image2, category_pair):
		if category_pair=='tb':
			return self.top_bottom_forward
		elif category_pair=='ts':
			return self.top_shoe_forward
		else:
			return self.bottom_shoe_forward

	def forward(self, objective, image1, image2, category_pair=None):
		if objective=='S':
			return self.similarity_forward(image1, image2)
		else:
			return self.compatibility_forward(image1, image2, compatibility_forward)


