import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
	"""
	Contrastive loss function.
	"""

	def __init__(self, margin_similar=1.0, margin_compatible=1.0):
		super(ContrastiveLoss, self).__init__()
		self.margin_similar = margin_similar
		self.margin_compatible = margin_compatible

	def forward(self, objective, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2)

		if objective=='S':
			margin = self.margin_similar
			coeff = 1.0
		else:
			margin = self.margin_compatible
			coeff = self.margin_similar/self.margin_compatible

		x1 = 1+(-1)*label
		x2 = torch.pow(euclidean_distance, 2)
		loss_contrastive = x1*x2
		x3 = label
		x4 = torch.pow(torch.clamp(margin + (-1)*euclidean_distance, min=0.0), 2)
		loss_contrastive += x3*x4
		loss_contrastive = torch.mean(loss_contrastive)

		return loss_contrastive
