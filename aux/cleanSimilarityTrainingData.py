import os
"""
This script removes all of the images in the rogueImages file from the files specified in fileName. 
"""

def iterateOverImages(fileName):
	with open('/data/srajpal2/AmazonDataset/similarity_training/rogueImages_Size.txt') as f:
		for line in f:
			removeImage(line.rstrip(), fileName)


def removeImage(url, fileName):
	os.system("grep -v '%s' %s >> /data/srajpal2/AmazonDataset/similarity_training/temp" % (url, fileName))
	os.system("mv /data/srajpal2/AmazonDataset/similarity_training/temp %s" % (fileName))

	print url

#iterateOverImages("/data/srajpal2/AmazonDataset/similarity_training/similarity_training_pairs.txt")
iterateOverImages("/data/srajpal2/AmazonDataset/training_images.json")
