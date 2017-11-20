import torch
from torch import optim
from similarity_network import Net
from hinge_loss import ContrastiveLoss
from torch.autograd import Variable
import os
from multiprocessing import Process, Pool
import torch.nn.functional as F
from torch.utils.data import DataLoader
from time import time
import torchvision.datasets as dset

# Hyperparameters
train_number_epochs = 1
version = 7
gradnorm_file = '../TrainingHistory/V%d/Similarity/grad_norm.txt' % (version)
trainingloss_file = '../TrainingHistory/V%d/Similarity/training_loss.txt' % (version)
infrequent_trainingloss_file = '../TrainingHistory/V%d/Similarity/infrequent_training_loss.txt' % (version)
valloss_file = '../TrainingHistory/V%d/Similarity/val_loss.txt' % (version)
image_size = 227
learning_rate = 0.005
primary_embedding_dim = 256
max_gradnorm = 40
training_datadir = '/data/srajpal2/AmazonDataset/similarity_training/minibatches/'
validation_data = '/data/srajpal2/AmazonDataset/pure_similarity_val_pairs.txt'
validation_evaluation_frequency = 200

if not os.path.isdir('/data/srajpal2/AmazonDataset/Checkpoints/V%d/Similarity' % version):
	os.makedirs('/data/srajpal2/AmazonDataset/Checkpoints/V%d/Similarity' % version)
if not os.path.isdir('../TrainingHistory/V%d/Similarity' % version):
	os.makedirs('../TrainingHistory/V%d/Similarity' % version)


os.environ["CUDA_VISIBLE_DEVICES"]="3"
net = Net(primary_embedding_dim, pretrained=True).cuda()
net.train()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)

training_history = open(trainingloss_file, 'w')
infreq_training_history = open(infrequent_trainingloss_file, 'w')

def perform_validation(checkpoint, iteration_num, prev_checkpoint):
	
	os.system("python similarity_validation.py %s %s %s %d %d %d %s" % (checkpoint, validation_data, valloss_file, image_size, primary_embedding_dim, iteration_num, prev_checkpoint))
	return

transform = Compose([])
mnist_dset = dset.MNIST(root="/data/srajpal2/MNIST/", train=True, transform=Compose([Resize(imageSize), RandomCrop(imageSize), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
loader = DataLoader(mnist_dset, num_workers=8, batch_size=128)

i_batch = -1
prev_time = time()
prev_checkpoint = None


for sample_batched in loader:
	im1 = sample_batched['im1']
	im2 = sample_batched['im2']
	label = sample_batched['label']

	im1, im2 , label = torch.squeeze(Variable(im1).cuda()), torch.squeeze(Variable(im2).cuda()), torch.squeeze(Variable(label.float()).cuda())

	output1, output2 = net(im1, im2)
	optimizer.zero_grad()
	loss_contrastive = criterion(output1,output2,label)
	loss_contrastive.backward()
	grad_norm = torch.nn.utils.clip_grad_norm(net.parameters(), max_gradnorm)
	optimizer.step()

	i_batch+=1

	if i_batch%validation_evaluation_frequency==0:
		infreq_training_history.write(str(i_batch) + ' ' + str(loss_contrastive.data[0]) + '\n')
		# Checkpoint Current Model
		torch.save({'minibatch':i_batch+1, 'state_dict': net.float().state_dict(), 'optimizer':optimizer.state_dict()}, "/data/srajpal2/AmazonDataset/Checkpoints/V%d/Similarity/minibatch%d.pth" % (version, i_batch+1))
		p = Process(target=perform_validation, args=("/data/srajpal2/AmazonDataset/Checkpoints/V%d/Similarity/minibatch%d.pth" % (version, i_batch+1), i_batch, prev_checkpoint))
		prev_checkpoint = "/data/srajpal2/AmazonDataset/Checkpoints/V%d/Similarity/minibatch%d.pth" % (version, i_batch+1)
		p.start()	
	print "Iter: %d, Loss: %.4f, GradNorm: %.2f, IterationTime: %.3f" % (i_batch, loss_contrastive.data[0], grad_norm, time()-prev_time)
	training_history.write("Iter: %d, Loss: %.4f, GradNorm: %.2f, IterationTime: %.3f\n" % (i_batch, loss_contrastive.data[0], grad_norm, time()-prev_time))
	prev_time = time()


training_history.close()
infreq_training_history.close()

