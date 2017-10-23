import json
import ast

class Trie:
	def __init__(self, data):
		self.data = data
		self.children = {}

class ManageTrie:

	def __init__(self):
		self.root = Trie('root')
	
	def add_to_trie(self, s):
		current_node = self.root
		count = 0
		for c in s:
			if c not in current_node.children.keys():
				new_child = Trie(c)
				current_node.children[c] = new_child
			current_node = current_node.children[c]
			count += 1
			if count==len(s):
				current_node.children["END"] = None		

	def lookup_in_trie(self, s):
		current_node = self.root
		found = True
		for c in s:
			if c not in current_node.children.keys():
				found = False
				break
			current_node = current_node.children[c]
		
		#return found
		return ("END" in current_node.children.keys())		
		
"""with open('test_ids.txt') as f:
	test_ids = f.readline()
	test_ids = [x[2:-1] for x in test_ids.rstrip().split(',')]
	trie = ManageTrie()
	for test_id in test_ids:
		trie.add_to_trie(test_id)

test_imUrls = []
test_asins = []

with open('meta_Clothing_Shoes_and_Jewelry.json') as f:
	for line in f:
		asin_beginning = line.find("'asin': ") + 9
		asin_end = line.find("'", asin_beginning)
		asin = line[asin_beginning:asin_end]

		if trie.lookup_in_trie(asin):

			imUrl_beginning = line.find("'imUrl': ") + 10
			imUrl_end = line.find("'", imUrl_beginning)
			imUrl = line[imUrl_beginning: imUrl_end]

			test_asins.append(asin)
			test_imUrls.append(imUrl)

with open('test_asins_imUrls.txt', 'w') as f:
	for i in xrange(len(test_asins)):
		f.write(test_asins[i] + ',' + test_imUrls[i] +'\n')"""
