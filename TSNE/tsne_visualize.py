from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

def create_tsne(loc_file, images_file):
	embeddings = np.load(loc_file)
	with open(images_file) as f:
		images = ['/'.join(x.rstrip().split('/')[-3:]) for x in f.readlines()]

	# print embeddings.shape, len(images), images[0]

	fig, ax = plt.subplots()
	#ax.plot(x, y)
	
	artists = []
	
	count = 0

	x_min = 100000
	y_min = 100000
	x_max = -100000
	y_max = -100000

	for loc,image in zip(embeddings, images):
		# loc = loc*1000
		try:
			image = Image.open(image)
			# image = image.resize((64, 64))
			#image = plt.imread(image)
		except TypeError:
			# Likely already an array...
			pass
		# im = OffsetImage(image, zoom=0.01, transform=ax.transAxes)
		# im.set_zoom(0.01)
		# im = OffsetImage(image)
		# ab = AnnotationBbox(im, (loc[0], loc[1]), xycoords='data', frameon=False)
		# print loc[1], loc[0]
		loc = loc*100
		# print (loc[0], loc[0]+150, loc[1], loc[1]+150)
		plt.imshow(image, extent=(loc[0], loc[0]+150, loc[1], loc[1]+150))

		if loc[0]-500 < x_min:
			x_min = loc[0] - 500
		if loc[0]+650 > x_max:
			x_max = loc[0] + 650
		if loc[1]-500 < y_min:
			y_min = loc[1] - 500
		if loc[1]+650 > y_max:
			y_max = loc[1] + 650


		# artists.append(ax.add_artist(ab))
		# break
		# count += 1
		# if count==10:
		# 	break

	axes = plt.gca()
	axes.set_xlim([min(x_min, y_min),max(x_max, y_max)])
	axes.set_ylim([min(x_min, y_min),max(x_max, y_max)])

	# ax.update_datalim(embeddings)
	# ax.autoscale()
	plt.show()
	return artists


create_tsne("Testing/tsne_locs.npy", "Testing/tsne_imgs.txt")
