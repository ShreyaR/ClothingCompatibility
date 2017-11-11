def removeSimilarityExamples(file_name):
	outfile = open('/'.join(file_name.split('/')[:-1] + ['pure_' + file_name.split('/')[-1]]),'w')
	
	with open(file_name) as f:
		for line in f:
			objective = line.rstrip().split(' ')[0]
			if objective == 'C':
				outfile.write(line)

	outfile.close()

removeSimilarityExamples('/data/srajpal2/AmazonDataset/fixed_testing_pairs.txt')
removeSimilarityExamples('/data/srajpal2/AmazonDataset/fixed_val_pairs.txt')
