import torch
from torch import optim
from hinge_loss import ContrastiveLoss
from minibatch_loader import dataset
from torch.autograd import Variable
from similarity_network import Net
import os
import sys
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"]="1"

class validation:

	def __init__(self, checkpoint, validation_data, valloss_file, image_size, primary_embedding_dim, iteration_num):
		self.validation_data = validation_data
		self.valloss_file = valloss_file
		self.auc_file = auc_file
		self.image_size = image_size
		self.iteration_num = iteration_num
		self.net = Net(primary_embedding_dim, pretrained=False).cuda()
		self.net.eval()
		# Load saved network
		# checkpoint = torch.load(checkpoint, map_location={'cuda:0':'cuda:1'})
		checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(0))
		self.net.load_state_dict(checkpoint['state_dict'])
		self.criterion = ContrastiveLoss()
		self.validation_loss()

	def validation_loss(self):
		data = dataset(self.validation_data, self.image_size)
		validation_dataloader = DataLoader(data, batch_size=128)
		val_history = open(self.valloss_file, 'a')
		losses = []

		for example in validation_dataloader.__getitem__():
	        	
        	im1 = example['im1']
        	im2 = example['im2']
        	label = example['label']

        	im1, im2 , label = Variable(im1).cuda(), Variable(im2).cuda() , Variable(label).cuda()
			output1,output2 = self.net(im1,im2)
            loss_contrastive = self.criterion(objective,output1,output2,label)
			
			losses.append(loss_contrastive.data[0])

	
		avg_loss = float(sum(losses))/len(losses)
		val_history.write("%d, %f\n" % (self.iteration_num, avg_loss))

        print "Validation Completed!"

validation(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))


