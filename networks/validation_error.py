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

	def __init__(self, checkpoint, validation_data, valloss_file, auc_file, image_size):
		self.validation_data = validation_data
		self.valloss_file = valloss_file
		self.auc_file = auc_file
		self.image_size = image_size
		self.gpu_num = gpu_num
		
		self.net = Net(primary_embedding_dim, sec_embedding_dim, pretrained=False).cuda()

		# Load saved network
		checkpoint = torch.load(checkpoint, map_location={'cuda:0':'cuda:1'})
		self.net.load_state_dict(checkpoint['state_dict'])
		self.validation_loss()


	def validation_loss(self):

		validation_dataloader = SiameseNetworkDataset(self.validation_data, self.image_size, 1)

		for example in train_dataloader.__getitem__():
	        objective = example[0]
	        im1 = example[1]
	        im2 = example[2]
	        label = example[3]

	        im1, im2 , label = Variable(im1).cuda(), Variable(im2).cuda() , Variable(label).cuda()
	        
	        if len(example)==5:
	            categorypair = example[4]
	            output1,output2 = net(objective,im1,im2,categorypair)
	        else:
	            output1,output2 = net(objective,im1,im2)

	        optimizer.zero_grad()
	        loss_contrastive = criterion(objective,output1,output2,label)

	        # Logging
	        grad_history.write(str(grad_norm) + '\n')
	        training_history.write(objective + ' ' + str(loss_contrastive) + '\n')


        print "Done training and Logging"
        print i, objective, loss_contrastive.data[0]

		



if i % validation_evaluation_frequency == 0:

	# Validation Loss
	loss = []
	for val_example in validation_dataloader.__getitem__():
		objective = val_example[0]
		im1 = val_example[1]
		im2 = val_example[2]
		label = val_example[3]
		im1, im2 , label = Variable(im1).cuda(), Variable(im2).cuda() , Variable(label).cuda()
		if len(val_example)==5:
			categorypair = val_example[4]
			output1,output2 = net(objective,im1,im2,categorypair)
		else:
			output1,output2 = net(objective,im1,im2)

		optimizer.zero_grad()
		loss_contrastive = criterion(objective,output1,output2,label)
		loss.append(loss_contrastive)

	avg_loss = float(sum(loss))/len(loss)
	val_history.write(str(avg_loss) + '\n')

	# print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
	iteration_number += validation_evaluation_frequency
	counter.append(iteration_number)
	loss_history.append(loss_contrastive.data[0])
	print "Done with validation set" 

	i+=1