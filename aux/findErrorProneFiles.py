from PIL import Image

with open('/data/srajpal2/AmazonDataset/fixed_val_pairs.txt') as f:
	count = 0
	for line in f:
		info = line.rstrip().split(' ')
		im1, im2 = info[1], info[2]
		try:
			Image.open(im1)
		except IOError:
			print im1
			count += 1
		try:
			Image.open(im2)
		except IOError:
			print im2
			count += 1

print count
