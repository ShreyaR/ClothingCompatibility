def removeStuff(fileName):
	outfile = '/'.join(fileName.split('/')[:-1] + ['fixed_' + fileName.split('/')[-1]])
	out = open(outfile, 'w')
	with open(fileName) as f:
		for line in f:
			out.write(' '.join(line.split(' ')[1:4]))
	out.close()
	return


removeStuff('/data/srajpal2/AmazonDataset/similarity_training/similarity_training_pairs.txt')		
