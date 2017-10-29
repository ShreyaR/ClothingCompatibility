import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class Net(torch.nn.Module):

	def __init__(self, embeddingSize):
		super(Net, self).__init__()
		self.vgg16 = vgg16(pretrained=True)
		self.vgg16.classifier = nn.Sequential(*(self.vgg16.classifier[i] for i in range(6)))
		self.vgg16.classifier.add_module('6', nn.Linear(4096, embeddingSize))

		# self.conv1 = nn.Conv2d(3, 64, 3) # Layer 1
		# self.conv2 = nn.Conv2d(64, 64, 3) # Layer 2
		# self.pool1 = nn.MaxPool2d(2)
		# self.conv3 = nn.Conv2d(64, 128, 3)
		# self.conv4 = nn.Conv2d(128, 128, 3)
		# self.pool2 = nn.MaxPool2d(2)
		# self.conv5 = nn.Conv2d(128, 256, 3)
		# self.conv6 = nn.Conv2d(256, 256, 3)
		# self.conv7 = nn.Conv2d(256, 256, 3)
		# self.pool3 = nn.MaxPool2d(2)
		# self.conv8 = nn.Conv2d(256, 512, 3)
		# self.conv9 = nn.Conv2d(512, 512, 3)
		# self.conv10 = nn.Conv2d(512, 512, 3)
		# self.pool4 = nn.MaxPool2d(2)
		# self.conv11 = nn.Conv2d(512, 512, 3)
		# self.conv12 = nn.Conv2d(512, 512, 3)
		# self.conv13 = nn.Conv2d(512, 512, 3)
		# self.pool5 = nn.MaxPool2d(2)
		# self.fc1 = nn.Linear(7*7*512, 4096)
		# self.fc2 = nn.Linear(4096, 4096)
		# self.fc3 = nn.Linear(4096, embeddingSize)

	def forward_once(self, x):
		"""
		Takes as input a 3*224*224 image, returns an embedding of length 4096.
		"""

		# x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
		# x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
		# x = self.pool3(F.relu(self.conv7(F.relu(self.conv6(F.relu(self.conv5(x)))))))
		# x = self.pool4(F.relu(self.conv10(F.relu(self.conv9(F.relu(self.conv8(x)))))))
		# x = self.pool5(F.relu(self.conv13(F.relu(self.conv12(F.relu(self.conv11(x)))))))
		# x = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))

		x = self.vgg16(x)

		return x

	def forward(self, image1, image2):
		"""
		Returns pair of embeddings for pair of training images.
		"""
		output1 = self.forward_once(image1)
		output2 = self.forward_once(image2)
		return output1, output2
