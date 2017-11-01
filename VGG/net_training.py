from torch import optim
from vgg import Net
from constrastive_loss import ContrastiveLoss
from dataset_loading import SiameseNetworkDataset
from torch.autograd import Variable
from torchvision.models import vgg16


# net = SiameseNetwork().cuda()
net = vgg16(pretrained=True)

class EncoderCNN(nn.Module):

    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.vgg = models.vgg16()
        self.vgg.load_state_dict(torch.load(vgg_checkpoint))
        self.vgg.classifier = nn.Sequential(
            *(self.vgg.classifier[i] for i in range(6)))

    def forward(self, images):
        return self.vgg(images)

criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)

counter = []
loss_history = [] 
iteration_number= 0

for epoch in range(0,Config.train_number_epochs):

    train_dataloader = SiameseNetworkDataset("/data/srajpal2/AmazonDataset/transformed_train.txt")
    for im1, im2, label in train_dataloader.__getitem__():

        im1, im2 , label = Variable(im1).cuda(), Variable(im2).cuda() , Variable(label).cuda()
        
        #Pre-trained net
        output1,output2 = net(im1,im2)
        optimizer.zero_grad()
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data[0])
show_plot(counter,loss_history)