from trie import ManageTrie
from random import uniform
import json

class traintestval_split:

	def __init__(self):
		self.main = '/data/srajpal2/AmazonData/updated_categories_meta_tbs.json'
		self.trainfile = '/data/srajpal2/AmazonData/training_images_intermediate.json'
		self.testfile = '/data/srajpal2/AmazonData/testing_images_intermediate.json'
		self.valfile = '/data/srajpal2/AmazonData/val_images_intermediate.json'
		self.training_imgs = ManageTrie()
		self.test_imgs = ManageTrie()
		self.val_imgs = ManageTrie()

		self.split_data()
		self.remove_cross_category_pairs()

	def split_data(self):
		trainfile = open(self.trainfile, 'w')
		testfile = open(self.testfile, 'w')
		valfile = open(self.valfile, 'w')

		with open(self.main) as f:
			for line in f:
				asin = json.loads(line.rstrip())['asin']
				prob = uniform()
				if 0<=prob<0.8:
					self.training_imgs.add_to_trie(asin)
					trainfile.write(line)
				elif 0.8<=prob<0.99:
					self.test_imgs.add_to_trie(asin)
					testfile.write(line)
				else:
					self.valfile.add_to_trie(asin)
					valfile.write(line)

		trainfile.close()
		testfile.close()
		valfile.close()

		return

	def remove_cross_category_pairs(self):
		trainoutput = '/data/srajpal2/AmazonData/training_images.json'
		testoutput = '/data/srajpal2/AmazonData/testing_images.json'
		valoutput = '/data/srajpal2/AmazonData/val_images.json'

		for trie, inputfile, outputfile in zip([self.training_imgs, self.test_imgs, self.val_imgs], [self.trainfile, self.testfile, self.valfile], [trainoutput, testoutput, valoutput]):
			outfile = open(outputfile, 'a')
			with open(inputfile) as f:
				for line in f:
					info = json.loads(line.rstrip())
					related = {}
					for k,v in info['related'].items():
						related[k] = []
						for item in v:
							if trie.lookup_in_trie(item):
								related[k].append(item)
					info["related"] = related
					json.dump(info, outfile)
					outfile.write('\n')
			outfile.close()

		return





