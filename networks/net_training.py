import torch
from torch import optim
# from vgg import Net
from alexnet import Net
from contrastive_loss import ContrastiveLoss
from minibatch_loading import SiameseNetworkDataset
from torch.autograd import Variable
import os
from multiprocessing import Process, Pool
#from validation_error import validation

# Hyperparameters
train_number_epochs = 1
minibatch_size = 32
gradnorm_file = '/data/srajpal2/AmazonDataset/TrainingHistory/grad_norm.txt'
auc_file = '/data/srajpal2/AmazonDataset/TrainingHistory/auc.txt'
trainingloss_file = '/data/srajpal2/AmazonDataset/TrainingHistory/training_loss.txt'
infrequent_trainingloss_file = '/data/srajpal2/AmazonDataset/TrainingHistory/infrequent_training_loss.txt'
valloss_file = '/data/srajpal2/AmazonDataset/TrainingHistory/val_loss.txt'
image_size = 227
learning_rate = 0.0005
primary_embedding_dim = 256
sec_embedding_dim = 32
max_gradnorm = 40
training_data = '/data/srajpal2/AmazonDataset/fixed_training_pairs.txt'
validation_data = '/data/srajpal2/AmazonDataset/fixed_val_pairs.txt'
validation_evaluation_frequency = 1000

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# net = SiameseNetwork().cuda()
net = Net(primary_embedding_dim, sec_embedding_dim, pretrained=True).cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)

counter = []
loss_history = []
iteration_number= 0
i=0

grad_history = open(gradnorm_file, 'w')
training_history = open(trainingloss_file, 'w')
val_history = open(valloss_file, 'w')
auc_history = open(auc_file, 'w')
infreq_training_history = open(infrequent_trainingloss_file, 'w')

def perform_validation(checkpoint, iteration_num):
	#val_instance = validation(checkpoint, validation_data, valloss_file, auc_file, image_size, primary_embedding_dim, sec_embedding_dim, iteration_num, learning_rate)
	
	os.system("python validation_error.py %s %s %s %s %d %d %d %d %f" % (checkpoint, validation_data, valloss_file, auc_file, image_size, primary_embedding_dim, sec_embedding_dim, iteration_num, learning_rate))
	return
		

for epoch in range(0,train_number_epochs):

    train_dataloader = SiameseNetworkDataset(training_data, image_size, minibatch_size)
    validation_dataloader = SiameseNetworkDataset(validation_data, image_size, 1)

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
        grad_norm = torch.nn.utils.clip_grad_norm(net.parameters(), max_gradnorm)
        loss_contrastive.backward()
        optimizer.step()

        # Logging
        grad_history.write(str(grad_norm) + '\n')
        training_history.write(objective + ' ' + str(loss_contrastive) + '\n')

        # print "Done training and Logging"
        #print i, objective, type(loss_contrastive.data[0])
	print i
        if i % validation_evaluation_frequency == 0:
		
		infreq_training_history.write(objective + ' ' + str(loss_contrastive.data[0]) + '\n')
            	# Checkpoint Current Model
            	torch.save({'epoch': epoch+1, 'minibatch':i+1, 'state_dict': net.float().state_dict(), 'optimizer':optimizer.state_dict()}, "/data/srajpal2/AmazonDataset/Checkpoints/epoch%d_minibatch%d.pth" % (epoch+1, i+1))
		#pool = Pool(processes=1)
		#pool.apply_async(perform_validation, ("/data/srajpal2/AmazonDataset/Checkpoints/epoch%d_minibatch%d.pth" % (epoch+1, i+1), i))
		p = Process(target=perform_validation, args=("/data/srajpal2/AmazonDataset/Checkpoints/epoch%d_minibatch%d.pth" % (epoch+1, i+1), i))
		p.start()
		#p.join(0.01)

	i+=1

grad_history.close()
training_history.close()
val_history.close()
auc_history.close()
infreq_training_history.close()


# show_plot(counter,loss_history)
