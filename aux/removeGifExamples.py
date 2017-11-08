class removeGIFs:

	def __init__(self, file):
		self.inputfile = file
		self.outputfile = file.split('/')[:-1] + "fixed_" + file.split('/')[-1]
		self.removeGifEgs()

	def removeGifEgs(self):
		outfile = open(self.outputfile, 'w')
		with open(self.inputfile) as f:
			for line in f:
				info = line.rstrip().split(' ')
				if info[1].split('.')!='gif' and info[2].split('.')!='gif':
					outfile.write(line)

		outfile.close()


train = removeGIFs('/data/srajpal2/AmazonDataset/training_pairs.txt')
test = removeGIFs('/data/srajpal2/AmazonDataset/testing_pairs.txt')
val = removeGIFs('/data/srajpal2/AmazonDataset/val_pairs.txt')



		