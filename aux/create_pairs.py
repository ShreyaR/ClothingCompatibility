import json
from random import choice

class pair_creation:

	def __init__(self, type_of_data, neg_to_pos):
	
		self.neg_to_pos = neg_to_pos
		self.type_of_data = type_of_data
		self.outfile = open('/data/srajpal2/AmazonDataset/%s_pairs.txt' % (self.type_of_data), 'w')
		self.category_map = {}
		self.url_map = {}
		self.inverse_category_maps = {'t':[], 'b':[], 's':[]}
		self.initial_data_pass()
		for k,v in self.inverse_category_maps.items():
			print k, len(v)
		
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
			for line in f:
				info = json.loads(line.rstrip())
				asin, img, related, cat = info["asin"], info["imUrl"], info["related"], info["category"]
				self.createPositiveNegativeExamples(asin, img, related, cat)
				count += 1
				if count==5:
					break

	def createPositiveNegativeExamples(self, asin, img, related, category):
		
		for asin2 in related['compatible']:
			# Add 1 positive compatibility example
			img2 = self.url_map[asin2]
			cat2 = self.category_map[asin2]
			cat_pair = ''.join(sorted([category, cat2]))
			self.outfile.write('C' + ' ' + img + ' ' + img2 + ' ' + '0' + ' ' + cat_pair + '\n')

		for asin2 in related['similar']:
			# Add 1 positive similarity example
			img2 = self.url_map[asin2]
			cat2 = self.category_map[asin2]
			cat_pair = ''.join(sorted([category, cat2]))
			self.outfile.write('S' + ' ' + img + ' ' + img2 + ' ' + '0' + '\n')

		negative_cat1, negative_cat2 = [x for x in ['t','b','s'] if x!=category]
		negative_compatible_cat1 = self.sampleChoice(len(self.inverse_category_maps[negative_cat1]), int(0.5*(self.neg_to_pos)*max(1, len(related['compatible']))), set(related['compatible']))
		negative_compatible_cat2 = self.sampleChoice(len(self.inverse_category_maps[negative_cat2]), int(0.5*(self.neg_to_pos)*max(1, len(related['compatible']))), set(related['compatible']))
		negative_similar = self.sampleChoice(len(self.inverse_category_maps[category]), self.neg_to_pos*max(1, len(related['similar'])), set(related['similar']))

		# Incompatible clothes from different categories to query category
		negative_catpair1 = ''.join(sorted([negative_cat1, category]))
		negative_catpair2 = ''.join(sorted([negative_cat2, category]))
		for i in negative_compatible_cat1:
			print negative_cat1, i
			asin2 = self.inverse_category_maps[negative_cat1][i]
			img2 = self.url_map[asin2]
			self.outfile.write('C' + ' ' + img + ' ' + img2 + ' ' + '1' + ' ' + negative_catpair1 + '\n')
		for i in negative_compatible_cat2:
			asin2 = self.inverse_category_maps[negative_cat1][i]
			img2 = self.url_map[asin2]
			self.outfile.write('C' + ' ' + img + ' ' + img2 + ' ' + '1' + ' ' + negative_catpair1 + '\n')

		# Dissimilar clothes from the same category as the query category
		for i in negative_similar:
			asin2 = self.inverse_category_maps[category][i]
			img2 = self.url_map[asin2]
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

x1 = pair_creation('training', 10)
x2 = pair_creation('testing', 2)
x3 = pair_creation('val', 2)




