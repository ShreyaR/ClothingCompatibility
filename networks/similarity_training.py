import torch
from torch import optim
from similarity_network import Net
from minibatch_loader import batched_dataset
from torch.autograd import Variable
import os
from multiprocessing import Process, Pool
import torch.nn.functional as F
from torch.utils.data import DataLoader
from time import time

# Hyperparameters
train_number_epochs = 1
version = 8
gradnorm_file = '../TrainingHistory/V%d/Similarity/grad_norm.txt' % (version)
trainingloss_file = '../TrainingHistory/V%d/Similarity/training_loss.txt' % (version)
infrequent_trainingloss_file = '../TrainingHistory/V%d/Similarity/infrequent_training_loss.txt' % (version)
valloss_file = '../TrainingHistory/V%d/Similarity/val_loss.txt' % (version)
image_size = 227
learning_rate = 0.005
primary_embedding_dim = 256
max_gradnorm = 40
training_datadir = '/data/srajpal2/AmazonDataset/Triplets/similarity_training/Training/'
validation_data = '/data/srajpal2/AmazonDataset/pure_similarity_val_pairs.txt'
validation_evaluation_frequency = 200

if not os.path.isdir('/data/srajpal2/AmazonDataset/Checkpoints/V%d/Similarity' % version):
	os.makedirs('/data/srajpal2/AmazonDataset/Checkpoints/V%d/Similarity' % version)
if not os.path.isdir('../TrainingHistory/V%d/Similarity' % version):
	os.makedirs('../TrainingHistory/V%d/Similarity' % version)


os.environ["CUDA_VISIBLE_DEVICES"]="3"
net = Net(primary_embedding_dim, pretrained=True).cuda()
net.train()
#criterion = ContrastiveLoss()
criterion = torch.nn.TripletMarginLoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)

training_history = open(trainingloss_file, 'w')
infreq_training_history = open(infrequent_trainingloss_file, 'w')

def perform_validation(checkpoint, iteration_num, prev_checkpoint):
	
	os.system("python similarity_validation.py %s %s %s %d %d %d %s" % (checkpoint, validation_data, valloss_file, image_size, primary_embedding_dim, iteration_num, prev_checkpoint))
	return

dset = batched_dataset(training_datadir, image_size)
loader = DataLoader(dset, num_workers=8)

i_batch = -1
prev_time = time()
prev_checkpoint = None


for sample_batched in loader:
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

	i_batch+=1

	if i_batch%validation_evaluation_frequency==0:
		infreq_training_history.write(str(i_batch) + ' ' + str(loss_triplet.data[0]) + '\n')
		# Checkpoint Current Model
		torch.save({'minibatch':i_batch+1, 'state_dict': net.float().state_dict(), 'optimizer':optimizer.state_dict()}, "/data/srajpal2/AmazonDataset/Checkpoints/V%d/Similarity/minibatch%d.pth" % (version, i_batch+1))
		p = Process(target=perform_validation, args=("/data/srajpal2/AmazonDataset/Checkpoints/V%d/Similarity/minibatch%d.pth" % (version, i_batch+1), i_batch, prev_checkpoint))
		prev_checkpoint = "/data/srajpal2/AmazonDataset/Checkpoints/V%d/Similarity/minibatch%d.pth" % (version, i_batch+1)
		p.start()	
	print "Iter: %d, Loss: %.4f, GradNorm: %.2f, IterationTime: %.3f" % (i_batch, loss_triplet.data[0], grad_norm, time()-prev_time)
	training_history.write("Iter: %d, Loss: %.4f, GradNorm: %.2f, IterationTime: %.3f\n" % (i_batch, loss_triplet.data[0], grad_norm, time()-prev_time))
	prev_time = time()


training_history.close()
infreq_training_history.close()

