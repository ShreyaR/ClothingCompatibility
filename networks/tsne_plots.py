import json
import os
from minibatch_loader import batched_dataset
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop, Compose
from torch.autograd import Variable
from similarity_network import Net
import torch
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.offsetbox import OffsetImage, AnnotationBbox


"""This script does everything for generating t-sne plots, only taking a json file as input.
Since we're making using of validation code to calculate embeddings (for efficiency), we
create triplets (although teh notation of anchor, pos, neg isn't relevant anymore). Therefore,
important that raw_json file has #lines = 3k, for any k (not too big).
"""

os.environ["CUDA_VISIBLE_DEVICES"]="1"

class GeneratePlots:
	
	def __init__(self, f, preprocess, primary_embedding_dim, checkpoint):
		if preprocess:
			self.tsne_dir = self.preprocess(f)
		else:
			self.tsne_dir = f

                self.net = Net(primary_embedding_dim, pretrained=False).cuda()
                self.net.eval()
                checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(0))
                self.net.load_state_dict(checkpoint['state_dict'])
		
		self.compute_embeddings(227)
		#self.dimensionality_reduction(5)
		#np.save("../TSNE/tsne_locs.npy", self.embedding_2D)
		
		#with open("../TSNE/tsne_imgs.txt","w") as f:
		#	for img in self.images:
		#		new_img = '~/Dropbox/V1_VGG/ClothingCompatibility/TSNE/Images/' + img.split('/')[-1]
  		#		f.write("%s\n" % new_img)

		for i in self.images:
			os.system("scp %s /data/srajpal2/AmazonDataset/tsne/Images/" % (i))
			
		
		#self.create_tsne()

		"""def create_tsne(self):
    		fig, ax = plt.subplots()
    		#ax.plot(x, y)
    		
    		artists = []
    		for loc,image in zip(self.embeddings_2D, self.images):
			print loc, loc[0], loc[1], image
			continue
            		try:
                        	image = plt.imread(image)
                	except TypeError:
                        	# Likely already an array...
                        	pass
			im = OffsetImage(image, zoom=0.1)
        		ab = AnnotationBbox(im, (loc[0], loc[1]), xycoords='data', frameon=False)
        		artists.append(ax.add_artist(ab))
    		ax.update_datalim(np.column_stack([x, y]))
    		ax.autoscale()
		plt.save()
    		return artists"""


	def compute_embeddings(self, imageSize):
				
                data_files = os.listdir(self.tsne_dir)
                data_files.sort()
                transform = Compose([Resize(imageSize), RandomCrop(imageSize), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


		flag = True

		for f in data_files:
			df = pd.read_csv(self.tsne_dir + f, delim_whitespace=True, header=None)
			im1 = df[0].apply(Image.open).apply(transform)
                	im1 = torch.stack([x for x in im1], 0)
                	im2 = df[1].apply(Image.open).apply(transform)
                	im2 = torch.stack([x for x in im2], 0)
                	im3 = df[2].apply(Image.open).apply(transform)
                	im3 = torch.stack([x for x in im3], 0)

			im1, im2, im3 = Variable(im1).cuda(), Variable(im2).cuda() , Variable(im3).cuda()
                        output1,output2,output3 = self.net(im1,im2,im3)
			
			images = df[0].tolist() + df[1].tolist() + df[2].tolist()
			embeddings = torch.cat((output1,output2,output3), dim=0)

			if flag:
				self.images = images
				self.embeddings = embeddings
				flag = False
			else:
				self.images += images
				self.embeddings = torch.cat((self.embeddings, embeddings), dim=0)

			print len(self.images), self.embeddings.size()

		return

	def dimensionality_reduction(self, manifold):
		np_embeddings = self.embeddings.data.cpu().numpy()
		self.embedding_2D = TSNE(n_components=2).fit_transform(np_embeddings)
		print self.embedding_2D
		return
		
	def preprocess(self, raw_json):
		"""
		raw_json: json file url
		"""
		
		tsne_outfile = '/'.join(raw_json.split('/')[:-1]+['tsne_triplets.txt'])
		tsne_dir = '/'.join(raw_json.split('/')[:-1]+['minibatches/'])
		if not os.path.isdir(tsne_dir):
			os.mkdir(tsne_dir)

		urls = []
		with open(raw_json) as f:
			for line in f:
				info = json.loads(line.rstrip())
				urls.append(info["imUrl"])
		
		urls_dict = {i:[] for i in range(3)}
		[urls_dict[i%3].append(urls[i]) for i in xrange(len(urls))]
		
		with open(tsne_outfile,'w') as f:
			for i in xrange(len(urls_dict[0])):
				f.write("%s %s %s\n" % (urls_dict[0][i], urls_dict[1][i], urls_dict[2][i]))

		os.system("split -l 64 -d %s %s" % (tsne_outfile, tsne_dir))
		return tsne_dir
		
GeneratePlots('/data/srajpal2/AmazonDataset/tsne/minibatches/', False, 256, '/data/srajpal2/AmazonDataset/tsne/V13.pth') 
