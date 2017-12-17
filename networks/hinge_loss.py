import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(torch.nn.Module):
	"""
	Contrastive loss function.
	"""

	def __init__(self, alpha=20):
		super(TripletLoss, self).__init__()
		self.alpha = alpha

	def forward(self, output1, output2, output3):

    		positive_dist = torch.pow((e1 - e2),2).sum(1)
    		negative_dist = torch.pow((e1 - e3),2).sum(1)
	    	overall_dist = positive_dist - negative_dist + alpha
    		per_triplet_max = torch.max(overall_dist, Variable(torch.FloatTensor([0.0])))
    		return per_triplet_max.sum()



		"""euclidean_distance = F.pairwise_distance(output1, output2)

		x1 = 1+(-1)*label
		x2 = torch.pow(euclidean_distance, 2)
	
		loss_contrastive = x1*x2
		x3 = label
		x4 = torch.pow(torch.clamp(self.margin + (-1.0)*euclidean_distance, min=0.0), 2)
		loss_contrastive += x3*x4
		loss_contrastive = torch.mean(loss_contrastive)

		return loss_contrastive"""
