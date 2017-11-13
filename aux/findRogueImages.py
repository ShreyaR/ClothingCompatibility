import json
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop, Compose
from time import time

transform = Compose([Resize(227), RandomCrop(227), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

out = open('/data/srajpal2/AmazonDataset/similarity_training/rogueImages_Size.txt', 'w')

with open('/data/srajpal2/AmazonDataset/similarity_training/uniqImages.txt') as f:
	prev_time = time()
	count = 0
	for line in f:
		try:
			im = transform(Image.open(line.rstrip()))
			if im.size()!=(3,227,227):
				out.write(line)
		except IOError:
			continue
		count += 1
		if count%10000==0:
			print count, time()-prev_time
