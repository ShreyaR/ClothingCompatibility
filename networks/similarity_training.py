import torch
from torch import optim
from similarity_network import Net
from hinge_loss import ContrastiveLoss
from minibacth_loader import batched_dataset
from torch.autograd import Variable
import os
from multiprocessing import Process, Pool
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Hyperparameters
train_number_epochs = 1
gradnorm_file = '/data/srajpal2/AmazonDataset/TrainingHistory/V1/grad_norm.txt'
trainingloss_file = '/data/srajpal2/AmazonDataset/TrainingHistory/V1/training_loss.txt'
infrequent_trainingloss_file = '/data/srajpal2/AmazonDataset/TrainingHistory/V1/infrequent_training_loss.txt'
valloss_file = '/data/srajpal2/AmazonDataset/TrainingHistory/V1/val_loss.txt'
image_size = 227
learning_rate = 0.0001
primary_embedding_dim = 256
max_gradnorm = 40
training_datadir = '/data/srajpal2/AmazonDataset/similarity_training/minibatches'
# validation_data = '/data/srajpal2/AmazonDataset/pure_fixed_val_pairs.txt'
validation_evaluation_frequency = 100

os.environ["CUDA_VISIBLE_DEVICES"]="0"
net = Net(primary_embedding_dim, pretrained=True).cuda()
net.train()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)

counter = []
loss_history = []
iteration_number= 0
i=0

training_history = open(trainingloss_file, 'w')
infreq_training_history = open(infrequent_trainingloss_file, 'w')

def perform_validation(checkpoint, iteration_num):
	
	os.system("python similarity_validation.py %s %s %s %d %d %d" % (checkpoint, validation_data, valloss_file, image_size, primary_embedding_dim, iteration_num))
	return

dset = batched_dataset(training_datadir, image_size)
loader = DataLoader(dset, num_workers=8)


for i_batch, sample_batched in enumerate(loader):
	im1 = sample_batched['im1']
	im2 = sample_batched['im2']
	label = sample_batched['label']

	im1, im2 , label = Variable(im1).cuda(), Variable(im2).cuda(), Variable(label).cuda()
	output1, output2 = net(im1, im2)
	optimizer.zero_grad()
	loss_contrastive = criterion(output1,output2,label)
	grad_norm = torch.nn.utils.clip_grad_norm(net.parameters(), max_gradnorm)
	loss_contrastive.backward()
	optimizer.step()
	print i_batch, loss_contrastive.data[0], grad_norm
	training_history.write(str(i) + ' ' + str(loss_contrastive.data[0]) + '\n')

	if i_batch%validation_evaluation_frequency==0:
		infreq_training_history.write(objective + ' ' + str(loss_contrastive.data[0]) + '\n')
		# Checkpoint Current Model
		torch.save({'epoch': epoch+1, 'minibatch':i+1, 'state_dict': net.float().state_dict(), 'optimizer':optimizer.state_dict()}, "/data/srajpal2/AmazonDataset/Checkpoints/epoch%d_minibatch%d.pth" % (epoch+1, i+1))
		p = Process(target=perform_validation, args=("/data/srajpal2/AmazonDataset/Checkpoints/epoch%d_minibatch%d.pth" % (epoch+1, i+1), i))
		p.start()		

training_history.close()
infreq_training_history.close()

