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

		if objective='similar':
			margin = self.margin_similar
		else:
			margin = self.margin_compatible

		loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
			(label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))

		return loss_contrastive