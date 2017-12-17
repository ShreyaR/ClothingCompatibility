import torch
from torch import optim
from hinge_loss import ContrastiveLoss
from minibatch_loader import dataset
from torch.autograd import Variable
from similarity_network import Net
import os
import sys
from torch.utils.data import DataLoader
import cPickle as pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

class validation:

	def __init__(self, checkpoint, validation_data, valloss_file, image_size, primary_embedding_dim, iteration_num, prev_checkpoint):
		self.validation_data = validation_data
		self.valloss_file = valloss_file
		self.image_size = image_size
		self.iteration_num = iteration_num
		self.net = Net(primary_embedding_dim, pretrained=False).cuda()
		self.net.eval()
		checkpoint_path = checkpoint
		# Load saved network
		checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(0))
		self.net.load_state_dict(checkpoint['state_dict'])
		self.criterion = torch.nn.TripletMarginLoss()
		loss = self.validation_loss()
		if prev_checkpoint!='None':
			prev_loss = pickle.load(open('/'.join(checkpoint_path.split('/')[:-2] + ['prev_loss.p']), 'rb'))
			if loss < prev_loss:
				pickle.dump(loss, open('/'.join(checkpoint_path.split('/')[:-2] + ['prev_loss.p']), 'wb'))
				os.system("rm %s" % prev_checkpoint)
			else:
				os.system("mv %s %s" % (prev_checkpoint, checkpoint_path))
		else:
			pickle.dump(loss, open('/'.join(checkpoint_path.split('/')[:-2] + ['prev_loss.p']), 'wb'))
		
			

	def validation_loss(self):
		data = dataset(self.validation_data, self.image_size)
		validation_dataloader = DataLoader(data, batch_size=128)
		val_history = open(self.valloss_file, 'a')
		losses = []

		for example in validation_dataloader:
	        	
        		im1 = example['im1']
        		im2 = example['im2']
        		im3 = example['im3']

        		im1, im2 , im3 = Variable(im1).cuda(), Variable(im2).cuda() , Variable(im3).cuda()
			output1,output2,output3 = self.net(im1,im2,im3)
       			loss_triplet = self.criterion(output1,output2,label)
			
			losses.append(loss_triplet.data[0])


		avg_loss = float(sum(losses))/len(losses)
		val_history.write("%d, %f\n" % (self.iteration_num, avg_loss))

        	print "Validation Completed!"
		return avg_loss

validation(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])


