import json
from random import choice, uniform
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
		idp = time()
		self.initial_data_pass()
		final_idp = idp - time()
		print "Final IDP", final_idp
		cp = time()
		self.create_pairs()
		final_cp = time() - cp
		print "Final CP", final_cp
	
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
			sumt1, sumt2, sumt3, sumt4 = 0,0,0,0
			for line in f:
				info = json.loads(line.rstrip())
				asin, img, related, cat = info["asin"], info["imUrl"], info["related"], info["category"]
				t1, t2, t3, t4 = self.createPositiveNegativeExamples(asin, img, related, cat)
				sumt1+=t1
				sumt2+=t2
				sumt3+=t3
				sumt4+=t4
				count += 1
				# if count==1000:
					# print count
				if count==50:
					print sumt1, sumt2, sumt3, sumt4
					break

	def createPositiveNegativeExamples(self, asin, img, related, category):
		

		t1 = time()
		for asin2 in related['compatible']:
			# Add 1 positive compatibility example
			img2 = self.url_map[asin2]
			cat2 = self.category_map[asin2]
			cat_pair = ''.join(sorted([category, cat2]))
			self.outfile.write('C' + ' ' + img + ' ' + img2 + ' ' + '0' + ' ' + cat_pair + '\n')
		final_t1 = time() - t1

		t2 = time()
		for asin2 in related['similar']:
			# Add 1 positive similarity example
			img2 = self.url_map[asin2]
			cat2 = self.category_map[asin2]
			cat_pair = ''.join(sorted([category, cat2]))
			self.outfile.write('S' + ' ' + img + ' ' + img2 + ' ' + '0' + '\n')
		final_t2 = time() - t2


		negative_cat1, negative_cat2 = [x for x in ['t','b','s'] if x!=category]
		cointoss = uniform(0,1)
		if cointoss<0.5:
			cat1_samples = self.compatibility_neg2pos*max(1, len(related['compatible']))/2
			cat2_samples = self.compatibility_neg2pos*max(1, len(related['compatible'])) - cat1_samples
		else:
			cat2_samples = self.compatibility_neg2pos*max(1, len(related['compatible']))/2
			cat1_samples = self.compatibility_neg2pos*max(1, len(related['compatible'])) - cat2_samples


		t3 = time()
		negative_compatible_cat1 = self.sampleChoice(len(self.inverse_category_maps[negative_cat1]), cat1_samples, set(related['compatible']))
		negative_compatible_cat2 = self.sampleChoice(len(self.inverse_category_maps[negative_cat2]), cat2_samples, set(related['compatible']))
		negative_similar = self.sampleChoice(len(self.inverse_category_maps[category]), self.similarity_neg2pos*max(1, len(related['similar'])), set(related['similar']))
		final_t3 = time() - t3

		# Incompatible clothes from different categories to query category
		t4 = time()
		negative_catpair1 = ''.join(sorted([negative_cat1, category]))
		negative_catpair2 = ''.join(sorted([negative_cat2, category]))
		for i in negative_compatible_cat1:
			asin2 = self.inverse_category_maps[negative_cat1][i]
			img2 = self.url_map[asin2]
			self.outfile.write('C' + ' ' + img + ' ' + img2 + ' ' + '1' + ' ' + negative_catpair1 + '\n')
		for i in negative_compatible_cat2:
			asin2 = self.inverse_category_maps[negative_cat2][i]
			img2 = self.url_map[asin2]
			self.outfile.write('C' + ' ' + img + ' ' + img2 + ' ' + '1' + ' ' + negative_catpair2 + '\n')

		# Dissimilar clothes from the same category as the query category
		for i in negative_similar:
			asin2 = self.inverse_category_maps[category][i]
			img2 = self.url_map[asin2]
			self.outfile.write('S' + ' ' + img + ' ' + img2 + ' ' + '1' + '\n')
		final_t4 = time() - t4

		return final_t1, final_t2, final_t3, final_t4

	def sampleChoice(self, n, k, collision_set):

		sampled_items = set()

		for i in xrange(k):
			while(True):
				x = choice(range(n))
				if x not in collision_set and x not in sampled_items:
					sampled_items.add(x)
					break

		return list(sampled_items)
print "Training Pairs"
x1 = pair_creation('training', 10, 10)
print "Testing Pairs"
x2 = pair_creation('testing', 1, 0)
print "Validation Pairs"
x3 = pair_creation('val', 1, 0)
