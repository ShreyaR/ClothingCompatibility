import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet
import numpy as np

class Net(torch.nn.Module):

	def __init__(self, primaryEmbeddingSize, secEmbeddingSize, pretrained=True):
		super(Net, self).__init__()
		self.network = alexnet(pretrained=pretrained)
		# self.network.classifier = nn.Sequential(*(self.network.classifier[i] for i in range()))
		self.network.classifier.add_module('7', nn.Linear(1000, primaryEmbeddingSize))

		self.bt_top = nn.Linear(primaryEmbeddingSize, secEmbeddingSize, bias=False)
		self.bt_bottom = nn.Linear(primaryEmbeddingSize, secEmbeddingSize, bias=False)
		self.st_shoe = nn.Linear(primaryEmbeddingSize, secEmbeddingSize, bias=False)
		self.st_top = nn.Linear(primaryEmbeddingSize, secEmbeddingSize, bias=False)
		self.bs_bottom = nn.Linear(primaryEmbeddingSize, secEmbeddingSize, bias=False)
		self.bs_shoe = nn.Linear(primaryEmbeddingSize, secEmbeddingSize, bias=False)
		
		for l in [self.bt_top, self.bt_bottom, self.st_shoe, self.st_top, self.bs_bottom, self.bs_shoe]:
			self.weights_init(l, 0.01)
	
	def weights_init(self, m, gain):
    		classname = m.__class__.__name__
    		if classname.find('Conv') != -1:
        		weight_shape = list(m.weight.data.size())
        		fan_in = np.prod(weight_shape[1:4])
        		fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        		w_bound = gain*np.sqrt(6. / (fan_in + fan_out))
		        m.weight.data.uniform_(-w_bound, w_bound)
		        if m.bias is not None:
				m.bias.data.fill_(0)
		elif classname.find('Linear') != -1:
        		weight_shape = list(m.weight.data.size())
        		fan_in = weight_shape[1]
        		fan_out = weight_shape[0]
        		w_bound = gain*np.sqrt(6. / (fan_in + fan_out))
        		m.weight.data.uniform_(-w_bound, w_bound)
			if m.bias is not None:
        			m.bias.data.fill_(0)


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
		# image1 = image1.unsqueeze(0)
		# image2 = image2.unsqueeze(0)
		if objective=='S':
			return self.similarity_forward(image1, image2)
		else:
			return self.compatibility_forward(image1, image2, category_pair)


