import json
from random import uniform
from numpy.random import choice
from time import time

class pair_creation:

	def __init__(self, type_of_data, compatibility_neg2pos, similarity_neg2pos):
	
		self.compatibility_neg2pos = compatibility_neg2pos
		self.similarity_neg2pos = similarity_neg2pos
		self.type_of_data = type_of_data
		self.outfile = open('/data/srajpal2/AmazonDataset/%s_pairs.txt' % (self.type_of_data), 'w')
		self.category_map = {}
		self.url_map = {}
		self.inverse_category_maps = {'t':[], 'b':[], 's':[]}
		self.initial_data_pass()
		self.create_pairs()
	
	def initial_data_pass(self):

		with open("/data/srajpal2/AmazonDataset/%s_images.json" % self.type_of_data) as f:
			for line in f:
				info = json.loads(line.rstrip())
				self.category_map[info["asin"]] = info["category"]
				self.inverse_category_maps[info["category"]].append(info["asin"]) 
				self.url_map[info["asin"]] = info["imUrl"]

	def create_pairs(self):

		with open("/data/srajpal2/AmazonDataset/%s_images.json" % self.type_of_data) as f:
			count = 0
			init_time = time()
			for line in f:
				info = json.loads(line.rstrip())
				asin, img, related, cat = info["asin"], info["imUrl"], info["related"], info["category"]
				self.createPositiveNegativeExamples(asin, self.imgUrlTransform(img), related, cat)
				count += 1
				if count==1000:
					print count, time()-init_time

	def createPositiveNegativeExamples(self, asin, img, related, category):
		

		for asin2 in related['compatible']:
			# Add 1 positive compatibility example
			img2 = self.imgUrlTransform(self.url_map[asin2])
			cat2 = self.category_map[asin2]
			cat_pair = ''.join([category, cat2])
			self.outfile.write('C' + ' ' + img + ' ' + img2 + ' ' + '0' + ' ' + cat_pair + '\n')

		for asin2 in related['similar']:
			# Add 1 positive similarity example
			img2 = self.imgUrlTransform(self.url_map[asin2])
			cat2 = self.category_map[asin2]
			self.outfile.write('S' + ' ' + img + ' ' + img2 + ' ' + '0' + '\n')

		negative_cat1, negative_cat2 = [x for x in ['t','b','s'] if x!=category]
		cointoss = uniform(0,1)
		if cointoss<0.5:
			cat1_samples = self.compatibility_neg2pos*max(1, len(related['compatible']))/2
			cat2_samples = self.compatibility_neg2pos*max(1, len(related['compatible'])) - cat1_samples
		else:
			cat2_samples = self.compatibility_neg2pos*max(1, len(related['compatible']))/2
			cat1_samples = self.compatibility_neg2pos*max(1, len(related['compatible'])) - cat2_samples


		negative_compatible_cat1 = choice(len(self.inverse_category_maps[negative_cat1]), cat1_samples)
		negative_compatible_cat2 = choice(len(self.inverse_category_maps[negative_cat2]), cat2_samples)
		negative_similar = choice(len(self.inverse_category_maps[category]), self.similarity_neg2pos*max(1, len(related['similar'])))

		# Incompatible clothes from different categories to query category
		negative_catpair1 = ''.join([category, negative_cat1])
		negative_catpair2 = ''.join([category, negative_cat2])
		for i in negative_compatible_cat1:
			asin2 = self.inverse_category_maps[negative_cat1][i]
			img2 = self.imgUrlTransform(self.url_map[asin2])
			self.outfile.write('C' + ' ' + img + ' ' + img2 + ' ' + '1' + ' ' + negative_catpair1 + '\n')
		for i in negative_compatible_cat2:
			asin2 = self.inverse_category_maps[negative_cat2][i]
			img2 = self.imgUrlTransform(self.url_map[asin2])
			self.outfile.write('C' + ' ' + img + ' ' + img2 + ' ' + '1' + ' ' + negative_catpair2 + '\n')

		# Dissimilar clothes from the same category as the query category
		for i in negative_similar:
			asin2 = self.inverse_category_maps[category][i]
			img2 = self.imgUrlTransform(self.url_map[asin2])
			self.outfile.write('S' + ' ' + img + ' ' + img2 + ' ' + '1' + '\n')

		return

	def sampleChoice(self, n, k, collision_set):

		sampled_items = set()

		for i in xrange(k):
			while(True):
				x = choice(range(n))
				if x not in collision_set and x not in sampled_items:
					sampled_items.add(x)
					break

		return list(sampled_items)


	def imgUrlTransform(self, url):
		url = url.split('/')
		url[3] = url[3] + 'set'
		return '/'.join(url)


print "Training Pairs"
x1 = pair_creation('training', 10, 10)
print "\n\nTesting Pairs"
x2 = pair_creation('testing', 1, 0)
print "\n\nValidation Pairs"
x3 = pair_creation('val', 1, 0)
