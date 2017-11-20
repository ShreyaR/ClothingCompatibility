import os
import struct
import numpy as np
from torch.utils.data import Dataset

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

class SiameseMNIST(Dataset):

	def __init__(self,dataset="training", path="."):

		if dataset is "training":
			fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        		fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
		elif dataset is "testing":
        		fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        		fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    		else:
        		raise ValueError, "dataset must be 'testing' or 'training'"

    		# Load everything in some numpy arrays
    		with open(fname_lbl, 'rb') as flbl:
        		magic, num = struct.unpack(">II", flbl.read(8))
        		lbl = np.fromfile(flbl, dtype=np.int8)

    		with open(fname_img, 'rb') as fimg:
        		magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        		img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

		classifiedImages = {}

		count = 0
		classes = np.unique(lbl)
		for c in classes:
			classifiedImages[c] = img[lbl==c]
			count += classifiedImages[c].shape[0]

		del img, lbl

		self.im1 = []
		self.im2 = []
		self.lbl = []

		for c in classifiedImages.keys():
			for im in classifiedImages[c]:
				for k in classifiedImages.keys():
					self.im1.append(im)
					self.im2.append(classifiedImages[k][np.random.randint(classifiedImages[k].shape[0])])
					if k==c:
						self.lbl.append(0)
					else:
						self.lbl.append(1)

		del classifiedImages
	
		self.im1 = np.array(self.im1)
		self.im2 = np.array(self.im2)
		self.lbl = np.array(self.lbl)

		randSequence = np.arange(self.lbl.size)
		np.random.shuffle(self.randSequence)	
		randSequence = randSequence.tolist()

		self.im1 = self.im1[randSequence, :, :]
		self.im2 = self.im2[randSequence, :, :]
		self.lbl = self.lbl[randSequence]
		
		self.transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	def __len__(self):
		return self.lbl.size

	def __getitem__(self, idx):
		im1 = self.transform(self.im1[idx,:,:])
		im2 = self.transform(self.im2[idx,:,:])
		lbl = float(self.lbl[idx])
		return {'im1':im1, 'im2':im2, 'label':lbl}



