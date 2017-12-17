import torch
from torch import optim
from similarity_network import Net
from hinge_loss import ContrastiveLoss
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from time import time
from PIL import Image
from torchvision.transforms import Resize, RandomCrop, Normalize, ToTensor, Compose
from clip_grad_debug import clip_grad_debug 
import os
import numpy as np

# Hyperparameters
train_number_epochs = 1
version = 6
gradnorm_file = '/data/srajpal2/AmazonDataset/TrainingHistory/V%d/Similarity/grad_norm.txt' % (version)
trainingloss_file = '/data/srajpal2/AmazonDataset/TrainingHistory/V%d/Similarity/training_loss.txt' % (version)
infrequent_trainingloss_file = '/data/srajpal2/AmazonDataset/TrainingHistory/V%d/Similarity/infrequent_training_loss.txt' % (version)
valloss_file = '/data/srajpal2/AmazonDataset/TrainingHistory/V%d/Similarity/val_loss.txt' % (version)
imageSize = 227
learning_rate = 0.00005
primary_embedding_dim = 256
max_gradnorm = 40
training_datadir = '/data/srajpal2/AmazonDataset/similarity_training/minibatches/'
validation_data = '/data/srajpal2/AmazonDataset/pure_similarity_val_pairs.txt'
validation_evaluation_frequency = 200

os.environ["CUDA_VISIBLE_DEVICES"]="3"
net = Net(primary_embedding_dim, pretrained=True).cuda()
net.train()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)

i_batch = -1
prev_time = time()
prev_checkpoint = None


transform = Compose([Resize(imageSize), RandomCrop(imageSize), ToTensor(), 
	Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

while(True):

	im1 = transform(Image.open('cat.jpg'))
	im2 = transform(Image.open('cat3.jpg'))
	label = torch.from_numpy(np.array([1]))


	im1, im2 , label = torch.unsqueeze(Variable(im1).cuda(), 0), torch.unsqueeze(Variable(im2).cuda(), 0), Variable(label.float()).cuda()

	output1, output2 = net(im1, im2)
	optimizer.zero_grad()
	loss_contrastive = criterion(output1,output2,label)
	loss_contrastive.backward()
	grad_norm = clip_grad_debug(net.parameters(), max_gradnorm)
	print grad_norm
	optimizer.step()

	i_batch+=1

	prev_time = time()

	break

