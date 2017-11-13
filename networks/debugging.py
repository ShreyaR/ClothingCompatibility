from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop, Compose
import pandas as pd
from torch import stack, from_numpy
import torch

transform = Compose([Resize(227), RandomCrop(227), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def __getitem__(file_name):
	# return self.load_file(self.data_files[idx])
	df = pd.read_csv(file_name, delim_whitespace=True, header=None)
       	im1 = df[0].apply(Image.open).apply(transform)
        #im1 = stack([x for x in im1 if x.size()==(3, 227, 227)], 0)
	print [x.size() for x in im1]
	im1 = stack([x for x in im1], 0)
        im2 = df[1].apply(Image.open).apply(transform)
        #im2 = stack([x for x in im2 if x.size()==(3, 227, 227)], 0)
        print [x.size() for x in im2]
	im2 = stack([x for x in im2], 0)
        label = torch.from_numpy(df[2].values)


        sample = {'im1': im1, 'im2':im2, 'label':label}
        return sample

__getitem__('/data/srajpal2/AmazonDataset/similarity_training/minibatches/9185')
