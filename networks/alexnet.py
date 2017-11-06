import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

class Net(torch.nn.Module):

	def __init__(self, primaryEmbeddingSize, secEmbeddingSize):
		super(Net, self).__init__()
		self.network = alexnet(pretrained=True)
		# self.network.classifier = nn.Sequential(*(self.network.classifier[i] for i in range()))
		self.network.classifier.add_module('7', nn.Linear(4096, primaryEmbeddingSize))

		self.bt_top = nn.Linear(primaryEmbeddingSize, secEmbeddingSize)
		self.bt_bottom = nn.Linear(primaryEmbeddingSize, secEmbeddingSize)
		self.st_shoe = nn.Linear(primaryEmbeddingSize, secEmbeddingSize)
		self.st_top = nn.Linear(primaryEmbeddingSize, secEmbeddingSize)
		self.bs_bottom = nn.Linear(primaryEmbeddingSize, secEmbeddingSize)
		self.bs_shoe = nn.Linear(primaryEmbeddingSize, secEmbeddingSize)


	def similarity_forward_once(self, x):
		"""
		Takes as input a 3*224*224 image, returns an embedding of length 4096.
		"""
		x = self.network(x)
		return x

	def similarity_forward(self, image1, image2):
		"""
		Returns pair of embeddings for pair of training images.
		"""
		output1 = self.similarity_forward_once(image1)
		output2 = self.similarity_forward_once(image2)
		return output1, output2

	def bt_forward_top(self, x):
		x = self.bt_top(self.network(x))
		return x

	def bt_forward_bottom(self, x):
		x = self.bt_bottom(self.network(x))
		return x

	def st_forward_top(self, x):
		x = self.st_top(self.network(x))
		return x

	def st_forward_shoe(self, x):
		x = self.st_shoe(self.network(x))
		return x

	def bs_forward_bottom(self, x):
		x = self.bs_bottom(self.network(x))
		return x

	def bs_forward_shoe(self, x):
		x = self.bs_shoe(self.network(x))
		return x

	def compatibility_forward(self, image1, image2, category_pair):
		catpair2net = {}
		catpair2net['bt'] = (self.bt_forward_bottom, self.bt_forward_top)
		catpair2net['tb'] = (self.bt_forward_top, self.bt_forward_bottom)
		catpair2net['st'] = (self.st_forward_shoe, self.st_forward_top)
		catpair2net['ts'] = (self.st_forward_top, self.st_forward_shoe)
		catpair2net['bs'] = (self.bs_forward_bottom, self.bs_forward_shoe)
		catpair2net['sb'] = (self.bs_forward_shoe, self.bs_forward_bottom)

		output1 = catpair2net[category_pair][0](image1)
		output2 = catpair2net[category_pair][1](image2)
		return output1, output2

	def forward(self, objective, image1, image2, category_pair=None):
		image1 = image1.unsqueeze(0)
		image2 = image2.unsqueeze(0)
		if objective=='S':
			return self.similarity_forward(image1, image2)
		else:
			return self.compatibility_forward(image1, image2, category_pair)


