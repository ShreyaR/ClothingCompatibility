import os

def segregate(f1, f2, f3, f4, f5):
	outfile_c1 = open(f2, 'w')
	outfile_c2 = open(f3, 'w')
	outfile_c3 = open(f4, 'w')
	outfile_s = open(f5, 'w')

	file_map = {'bt':outfile_c1,
			'bs':outfile_c2,
			'st':outfile_c3}
	flip_list = {'tb', 'ts', 'sb'}

	print f1
	with open(f1) as f:
		count = 0
		for line in f:
			count += 1
			objective = line.rstrip().split(' ')[0]
			if objective=='C':
				category = line.rstrip().split(' ')[4]
				if category in flip_list:
					category = "%c%c" % (category[1], category[0])
					info = line.rstrip().split(' ')[1:4]
					info[1], info[0] = info[0], info[1]
					info = ' '.join(info)
				else:
					info = ' '.join(line.rstrip().split(' ')[1:4])
				file_map[category].write(info + '\n')
			else:
				info = ' '.join(line.rstrip().split(' ')[1:4])
				outfile_s.write(info + '\n')
	print count
	outfile_c1.close()
	outfile_c2.close()
	outfile_c3.close()
	outfile_s.close()

	return

def combine(f1, f2, f3, l1, l2):
	f4 = '/'.join(f3.split('/')[:-1] + ['temp_' + f3.split('/')[-1]])
	temp = open(f4, 'w')
	with open(f1) as f:
		for i in xrange(l1):
			temp.write(f.readline())

	with open(f2) as f:
		for i in xrange(l2):
			temp.write(f.readline())

	temp.close()
	os.system("shuf %s << %s; rm %s" % (f4, f3, f4))


		
segregate('/data/srajpal2/AmazonDataset/random_fixed_training_pairs.txt',
	'/data/srajpal2/AmazonDataset/bt_compatibility_training_pairs.txt',
	'/data/srajpal2/AmazonDataset/bs_compatibility_training_pairs.txt',
	'/data/srajpal2/AmazonDataset/st_compatibility_training_pairs.txt',
	'/data/srajpal2/AmazonDataset/similarity_training_pairs.txt')
