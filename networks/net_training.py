from torch import optim
# from vgg import Net
from alexnet import Net
from constrastive_loss import ContrastiveLoss
from dataset_loading import SiameseNetworkDataset
from torch.autograd import Variable
import os


os.environ["CUDA_VISIBLE_DEVICES"]="0"
# net = SiameseNetwork().cuda()
net = Net().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)

counter = []
loss_history = []
iteration_number= 0
train_number_epochs = 5

# for epoch in range(0,Config.train_number_epochs):

for epoch in range(0,train_number_epochs):
    # train_dataloader = SiameseNetworkDataset("/data/srajpal2/AmazonDataset/transformed_train.txt")
    train_dataloader = SiameseNetworkDataset("/data/srajpal2/AmazonDataset/training_pairs.txt")
    i=0
    for example in train_dataloader.__getitem__():

        objective = example[0]
        im1 = example[1]
        im2 = example[2]
        label = example[3]

        im1, im2 , label = Variable(im1).cuda(), Variable(im2).cuda() , Variable(label).cuda()
        
        if len(example==5):
            categorypair = example[5]
            output1,output2 = net(objective,im1,im2,label,categorypair)
        else:
            output1,output2 = net(objective,im1,im2,label)

        optimizer.zero_grad()
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        
        i+=1        

        if i % 10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data[0])

show_plot(counter,loss_history)