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
image_size = 227
learning_rate = 0.001
primary_embedding_dim = 256
max_gradnorm = 40
margin = 0.5
opt = "RMS"
minibatch_size = 64
training_datadir = '/data/srajpal2/AmazonDataset/Triplets/similarity_training/Training/'

os.environ["CUDA_VISIBLE_DEVICES"]="2"

dset = batched_dataset(training_datadir, image_size)
loader = DataLoader(dset, num_workers=30)

i_batch = 0
prev_time = time()


for sample_batched in loader:
	i_batch += 1
	print i_batch
	continue

