from trie import ManageTrie
import json
from datetime import datetime

pathToMetaFile = "/data/srajpal2/meta_Clothing_Shoes_and_Jewelry.json"
pathToDeduplicatedFile = "/data/srajpal2/deducplicatedMeta.json"

idTrie = ManageTrie()

outfile = open(pathToDeduplicatedFile, 'w')

# titles = ['Women', 'Womens Lingerie Underwear Colorful C-String Thong Panty']

with open(pathToMetaFile) as f:
	count = 0
	for line in f:
		title_beginning = line.find("'imUrl': ") + 10
                title_end = line.find("'", title_beginning)
                title = line[title_beginning:title_end]
		#title = titles[count]
		#print title
		if idTrie.lookup_in_trie(title):
			continue
		idTrie.add_to_trie(title)
		outfile.write(line)
		count += 1
		if count%100000==0:
			print count/(1.0*1503384), datetime.now()
