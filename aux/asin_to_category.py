import json
import time
from test_ids_trie import Trie, ManageTrie

with open('test_ids.txt') as f:
        test_ids = f.readline()
        test_ids = [x[2:-1] for x in test_ids.rstrip().split(',')]
        trie = ManageTrie()
        for test_id in test_ids:
                trie.add_to_trie(test_id)

test_asins = []

class GetCategory:

	def __init__(self):
		self.categories = set()
		with open("categories.txt") as f:
			for line in f:
				self.categories.add(line.rstrip())
		self.fileList = {i:open("Categories/%s"%i, 'w') for i in self.categories}

	def getAllAnnotatedCategories(self, s):
		cats = set()
		for x in s[1:-1].split('], ['):
			for y in x.split(", '"):
				y = y.strip("'")
				if y in self.categories:
					cats.add(y)

		return cats

	def getCategories(self, catString, asin):
		cats = self.getAllAnnotatedCategories(catString)
		for c in cats:
			self.fileList[c].write(asin+'\n')
		return cats
		
		

categoryFinder = GetCategory()
print "Intialized categories."

outFile = open('related_items.json', 'a') 

with open('meta_Clothing_Shoes_and_Jewelry.json') as f:
	count = 0
        for line in f:

                asin_beginning = line.find("'asin': ") + 9
                asin_end = line.find("'", asin_beginning)
                asin = line[asin_beginning:asin_end]

                if trie.lookup_in_trie(asin):

                        categories_beginning = line.find("'categories': ") + 15
                        categories_end = line.find("']]", categories_beginning) + 2
                        categories = line[categories_beginning: categories_end]
			categories = categoryFinder.getCategories(categories,asin)

			copurchased1_beginning = line.find("'also_bought'") + 15
			copurchased1_end = line.find(']', copurchased1_beginning) + 1
			if copurchased1_beginning != 14:
				copurchased1 = line[copurchased1_beginning:copurchased1_end]
				copurchased1 = [x.strip("'") for x in copurchased1[1:-1].split(', ')]
			else:
				copurchased1 = []

			copurchased2_beginning = line.find("'bought_together'") + 19
                        copurchased2_end = line.find(']', copurchased2_beginning) + 1
                        if copurchased2_beginning != 18:
                                copurchased2 = line[copurchased2_beginning:copurchased2_end]
				copurchased2 = [x.strip("'") for x in copurchased2[1:-1].split(', ')]
			else:
				copurchased2 = []
			related = copurchased1 + copurchased2

			info = {'asin':asin, 'categories':list(categories), 'related':related}
			json.dump(info, outFile)
			outFile.write('\n')
		count +=1
		if count%10000==0:
			print count/15000.0, time.time()
					
