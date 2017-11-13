import os

def iterateOverImages():
	with open('/data/srajpal2/AmazonDataset/similarity_training/rogueImages_Size.txt') as f:
		for line in f:
			removeImage(line.rstrip())


def removeImage(url):
	os.system("grep -v '%s' /data/srajpal2/AmazonDataset/similarity_training/similarity_training_pairs.txt >> /data/srajpal2/AmazonDataset/similarity_training/temp" % (url))
	os.system("mv /data/srajpal2/AmazonDataset/similarity_training/temp /data/srajpal2/AmazonDataset/similarity_training/similarity_training_pairs.txt")

	print url

iterateOverImages()
