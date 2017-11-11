import torch
from torch import optim
# from vgg import Net
from alexnet import Net
from contrastive_loss import ContrastiveLoss
from minibatch_loading import SiameseNetworkDataset
from torch.autograd import Variable
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

class validation:

	def __init__(self, checkpoint, validation_data, valloss_file, auc_file, image_size, primary_embedding_dim, sec_embedding_dim, iteration_num, learning_rate):
		self.validation_data = validation_data
		self.valloss_file = valloss_file
		self.auc_file = auc_file
		self.image_size = image_size
		self.iteration_num = iteration_num
		self.net = Net(primary_embedding_dim, sec_embedding_dim, pretrained=False).cuda()
		self.net.eval()
		# Load saved network
		# checkpoint = torch.load(checkpoint, map_location={'cuda:0':'cuda:1'})
		checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(0))
		self.net.load_state_dict(checkpoint['state_dict'])
		self.criterion = ContrastiveLoss()
		self.optimizer = optim.Adam(self.net.parameters(),lr=learning_rate)
		self.validation_loss()


	def validation_loss(self):

		validation_dataloader = SiameseNetworkDataset(self.validation_data, self.image_size, 1)
		val_history = open(self.valloss_file, 'w')
		losses = []
		count = 0

		for example in validation_dataloader.__getitem__():
	        	objective = example[0]
	        	im1 = example[1]
	        	im2 = example[2]
	        	label = example[3]

	        	im1, im2 , label = Variable(im1).cuda(), Variable(im2).cuda() , Variable(label).cuda()
	        	
	        	if len(example)==5:
	            	
				categorypair = example[4]
	            		output1,output2 = self.net(objective,im1,im2,categorypair)
	        	else:
	            		output1,output2 = self.net(objective,im1,im2)

	        	self.optimizer.zero_grad()
	        	loss_contrastive = self.criterion(objective,output1,output2,label)
			losses.append(loss_contrastive)
	        	# Logging
		
		avg_loss = float(sum(losses))/len(losses)
		val_history.write("%d, %d\n" % (self.iteration_num, avg_loss))

        	print "Done"


validation("/data/srajpal2/AmazonDataset/Checkpoints/epoch1_minibatch1.pth", "/data/srajpal2/AmazonDataset/fixed_val_pairs.txt", "/data/srajpal2/AmazonDataset/TrainingHistory/val_history.txt", "whatevs", 227, 256, 32, 0, 0.0005)


