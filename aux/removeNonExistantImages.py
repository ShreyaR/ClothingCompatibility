from trie import ManageTrie
from time import time
tr = ManageTrie()

def findUniqueImages():
	out = open('/data/srajpal2/AmazonDataset/similarity_training/uniqImages.txt', 'w')
	with open('/data/srajpal2/AmazonDataset/similarity_training/similarity_training_pairs.txt') as f:
		count = 0
		prev_time = time()
		for line in f:
			im1, im2, _ = line.rstrip().split(' ')
			if not tr.lookup_in_trie(im1):
				out.write(im1 + '\n')
				tr.add_to_trie(im1)
			if not tr.lookup_in_trie(im2):
                        	out.write(im2 + '\n')
                        	tr.add_to_trie(im2)
			count += 1
			if count % 10000 ==0:
				print count, time() - prev_time
				prev_time = time()
	out.close()

findUniqueImages()	

