import json
from random import choice

outfile = open('/data/srajpal2/AmazonDataset/pairs.txt', 'w')

# category_map = {}
url_map = {}
inverse_category_maps = {'t':[], 'b':[], 's':[]}


with open("/data/srajpal2/AmazonDataset/updated_categories_meta_tbs.json") as f:
	for line in f:
		info = json.loads(line.rstrip())
		inverse_category_maps[info["category"]].append(info["asin"]) 
		url_map[info["asin"]] = info["url"]
		# asins.append(info["asin"])

with open("/data/srajpal2/AmazonDataset/updated_categories_meta_tbs.json") as f:
	for line in f:
		info = json.loads(line.rstrip())
		related = info["related"]

		for i in related['compatible']:
			


