import torch
from torch import optim
from similarity_network import Net
from minibatch_loader import batched_dataset, dataset
from torch.autograd import Variable
import os
import sys
from multiprocessing import Process, Pool
import torch.nn.functional as F
from torch.utils.data import DataLoader
from time import time

#####################################
#Hyperparameters

version = int(sys.argv[1])
opt = sys.argv[4]
train_number_epochs = 1
image_size = 227
learning_rate = float(sys.argv[2])
max_num_iter = int(sys.argv[3])
primary_embedding_dim = 256
max_gradnorm = 40
margin = 0.5
gpu_training = sys.argv[5]
minibatch_size = 64

#####################################
#Logs

valtrainingloss_file = '../TrainingHistory/TrainTestGaps/%s/val_training_loss_V%d.txt' % (opt, version)
info_file = '../TrainingHistory/TrainTestGaps/%s/info.txt' % (opt)

if not os.path.isdir('../TrainingHistory/TrainTestGaps/%s/' % opt):
	os.makedirs('../TrainingHistory/TrainTestGaps/%s/' % opt)

with open(info_file, 'w') as f:
	f.write("Learning Rate: %f\nPrimary Embedding Dimension: %d\nMargin: %f\nOptimizer: %s\nMinibatch Size: %d" % (learning_rate, primary_embedding_dim, margin, opt, minibatch_size))

val_training_history = open(valtrainingloss_file, 'w')

training_datadir = '/data/srajpal2/AmazonDataset/Triplets/similarity_training/Training/'
validation_data = '/data/srajpal2/AmazonDataset/Triplets/similarity_training/val_triplets.txt'


#####################################
#Network Info

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_training
net = Net(primary_embedding_dim, pretrained=False).cuda()
# net.train()
net.eval() ######### Removed dropout here
#criterion = ContrastiveLoss()
criterion = torch.nn.TripletMarginLoss(margin=margin)
if opt=='RMSprop':
	optimizer = optim.RMSprop(net.parameters(),lr=learning_rate)
if opt=='Adam':
	optimizer = optim.Adam(net.parameters(),lr=learning_rate)
if opt=='SGD':
	optimizer = optim.SGD(net.parameters(),lr=learning_rate)
if opt=='Adagrad':
	optimizer = optim.Adagrad(net.parameters(),lr=learning_rate)


#####################################
#Training & Validation Dataset Info

# Training Data
dset = batched_dataset(training_datadir, image_size)
loader = DataLoader(dset, num_workers=8)


# Validation Data
data = dataset(validation_data, image_size)
validation_dataloader = DataLoader(data, batch_size=325)

for example in validation_dataloader:
	val_im1 = example['im1']
	val_im2 = example['im2']
	val_im3 = example['im3']
	val_im1, val_im2 , val_im3 = Variable(val_im1).cuda(), Variable(val_im2).cuda() , Variable(val_im3).cuda()


#####################################
#Network Training

i_batch = -1
prev_time = time()
prev_checkpoint = None

corrupt_batches = 0

for sample_batched in loader:

	i_batch+=1
	
	im1 = sample_batched['im1']
	im2 = sample_batched['im2']
	im3 = sample_batched['im3']
	im1, im2 , im3 = torch.squeeze(Variable(im1).cuda()), torch.squeeze(Variable(im2).cuda()), torch.squeeze(Variable(im3).cuda())
	output1, output2, output3 = net(im1, im2, im3)

	optimizer.zero_grad()
	loss_triplet = criterion(output1,output2,output3)
	loss_triplet.backward()
	grad_norm = torch.nn.utils.clip_grad_norm(net.parameters(), max_gradnorm)
	optimizer.step()

	output1,output2,output3 = net(val_im1,val_im2,val_im3)
	val_loss_triplet = criterion(output1,output2,output3)
	val_training_history.write("%d, %.4f, %.4f\n" % (i_batch, loss_triplet.data[0], val_loss_triplet.data[0]))

	print "Iter: %d, Loss: %.4f, ValLoss: %.4f, IterationTime: %.3f" % (i_batch, loss_triplet.data[0], val_loss_triplet.data[0], time()-prev_time)
	prev_time = time()

	if i_batch==max_num_iter:
		break

val_training_history.close()

