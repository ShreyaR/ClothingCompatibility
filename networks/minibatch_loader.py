from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop, Compose
# import numpy as np
from torch import stack, from_numpy
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class batched_dataset(Dataset):
    
	def __init__(self, data_dir, imageSize):
		self.data_files = os.listdir('data_dir')
		self.transform = Compose([Resize(imageSize),
								  RandomCrop(imageSize),
								  ToTensor(),
								  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		sort(self.data_files)

	def __getindex__(self, idx):
		# return self.load_file(self.data_files[idx])
		df = pd.read_csv(self.data_files[idx], delim_whitespace=True, header=None)
		im1 = df[0].apply(Image.open).apply(self.transform)
		im1 = stack([x for x in im1], 0)
		im2 = df[1].apply(Image.open).apply(self.transform)
		im2 = stack([x for x in im2], 0)
		labels = torch.from_numpy(df[2].values)
		sample = {'im1': im1, 'im2':im2, 'label':label}
		return sample

	def __len__(self):
		return len(self.data_files)


class dataset(Dataset):
	def __init__(self, file_path, imageSize):
		
		self.df = pd.read_csv(file_path, delim_whitespace=True, header=None)
		self.transform = Compose([Resize(imageSize),
								  RandomCrop(imageSize),
								  ToTensor(),
								  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		
	def len(self):
		return len(self.df)

	def __getitem__(self, idx):

		im1 = self.transform(Image.open(self.df[0][idx]))
		im2 = self.transform(Image.open(self.df[1][idx]))
		label = float(self.df[2][idx])
		sample = {'im1': im1, 'im2':im2, 'label':label}
		return sample
		
