"""
This script is meant to preprocess the versions of training_images.json, testing_images.json and val_images.json.
This script currently does the work of the following scripts:
- cleanSimilarityTrainingData.py: script to remove rogueImages that don't have 3 channels
- removeGifExamples.py 
"""

import json
import os
from time import time

def data_preprocess(file_name, rogue_images):
	with open(rogue_images) as f:
		rogue = f.readlines()
	rogue = set([x.rstrip() for x in rogue])

	outfile = open('temp.txt', 'a')

	with open(file_name) as f:
		count = 0
		init_time = time()
		for line in f:
			count += 1
			if count==1000:
				print count, time()-init_time
			info = json.loads(line.rstrip())
			url = info["imUrl"]
			if url in rogue:
				continue
			if url.split('.')[-1] == 'gif':
				continue
			info["imUrl"] = url
			json.dump(info, outfile)

	os.system("mv temp.txt %s" % (file_name))
	return



def imgUrlTransform(url):
	"""
	Takes care of some minor formatting in img urls
	"""
	url = url.split('/')
	url[3] = url[3] + 'set'
	return '/'.join(url)