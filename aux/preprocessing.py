"""
This script is meant to preprocess the versions of training_images.json, testing_images.json and val_images.json.
This script currently does the work of the following scripts:
- cleanSimilarityTrainingData.py: script to remove rogueImages that don't have 3 channels
- removeGifExamples.py 
- also removes corrupted Imgs that exist in dataset but can't be opened by PIL
"""

import json
import os
from time import time
from trie import ManageTrie

tr = ManageTrie()

def data_preprocess(file_name, rogue_images, corrupted_images, AddSuffix=False, RemoveSuffix=False):
	

	with open(rogue_images) as f:
		rogue = f.readlines()
	with open(corrupted_images) as f:
		corrupted = f.readlines()
	rogue = set([x.rstrip() for x in rogue])
	corrupted = set([x.rstrip() for x in corrupted])
	outfile = open('temp.txt', 'a')

	asins_to_remove = set()

	with open(file_name) as f:
		count = 0
		init_time = time()
		for line in f:
			count += 1
			if count%1000==0:
				print count, time()-init_time
			info = json.loads(line.rstrip())
			url = info["imUrl"]
			if AddSuffix:
				url = add_suffix(url)
			if RemoveSuffix:
				url = remove_suffix(url)
			if url in rogue:
				continue
			if url.split('.')[-1] == 'gif':
				continue
			if info["asin"] in corrupted:
				continue
			tr.add_to_trie(info["asin"])
			info["imUrl"] = url
			json.dump(info, outfile)
			outfile.write('\n')
	outfile.close()
	os.system("mv temp.txt %s" % (file_name))
	os.system("rm temp.txt")
	
	print "Closing the set"
	outfile = open('temp.txt', 'a')
	with open(file_name) as f:
		count = 0
                init_time = time()
                for line in f:
                        count += 1
                        if count%1000==0:
                                print count, time()-init_time
                        
			info = json.loads(line.rstrip())
                        related = info["related"]
			new_related = {}
			for k,v in related.items():
				v_new = []
				for i in v:
					if tr.lookup_in_trie(i):
						v_new.append(i)
				#new_related[k] = list(set(v).difference(asins_to_remove))
				new_related[k] = v_new
			info["related"] = new_related
			json.dump(info, outfile)
                        outfile.write('\n')
        outfile.close()
        os.system("mv temp.txt %s" % (file_name))
	os.system("rm temp.txt")
	return

def add_suffix(s):
	l = s.split('/')
	l[3] = l[3] + 'set'
	return '/'.join(l)	

def remove_suffix(s):
        l = s.split('/')
        l[3] = l[3][:-3]
        return '/'.join(l)

#data_preprocess('/data/srajpal2/AmazonDataset/GoldStandard/training_images.json', 'rogueImages_Size.txt', 'corruptedImages.txt', True)
data_preprocess('/data/srajpal2/AmazonDataset/GoldStandard/val_images.json', 'rogueImages_Size.txt', "corruptedImages.txt")
data_preprocess('/data/srajpal2/AmazonDataset/GoldStandard/testing_images.json', 'rogueImages_Size.txt', "corruptedImages.txt")
