import json
from random import choice

outfile = open('/data/srajpal2/AmazonDataset/pairs.txt', 'w')

category_map = {}
url_map = {}
inverse_category_maps = {'t':[], 'b':[], 's':[]}


with open("/data/srajpal2/AmazonDataset/updated_categories_meta_tbs.json") as f:
	for line in f:
		info = json.loads(line.rstrip())
		category_map[info["asin"]] = info["category"]
		inverse_category_maps[info["category"]].append(info["asin"]) 
		url_map[info["asin"]] = info["url"]
		# asins.append(info["asin"])

with open("/data/srajpal2/AmazonDataset/updated_categories_meta_tbs.json") as f:
	count = 0
	for line in f:
		info = json.loads(line.rstrip())
		related = info["related"]
		img1 = info["imUrl"]
		cat1 = info["category"]

		for i in related['compatible']:

			# Add 1 positive compatibility example
			img2 = url_map[i]
			cat2 = category_map[i]
			cat_pair = ''.join(sorted([cat1, cat2]))
			outfile.write('C' + ' ' + img1 + ' ' + img2 + ' ' + '0' + ' ' + cat_pair + '\n')

			# Find 10 negative compatibility examples
			for c in inverse_category_maps.keys():
				if c==cat1:
					continue
				for j in choice(len(inverse_category_maps[c]), 5):

					asin2 = category_map[j]
					if asin2 in related['compatible']:
						continue

					cat2 = category_map[asin2]
					img2 = url_map[asin2]
					cat_pair = ''.join(sorted([cat1, cat2]))
					outfile.write('C' + ' ' + img1 + ' ' + img2 + ' ' + '1' + ' ' + cat_pair + '\n')

		if len(related['compatible'])==0:
			for c in inverse_category_maps.keys():
				if c==cat1:
					continue
				for j in choice(len(inverse_category_maps[c]), 5):

					asin2 = category_map[j]
					if asin2 in related['compatible']:
						continue

					cat2 = category_map[asin2]
					img2 = url_map[asin2]
					cat_pair = ''.join(sorted([cat1, cat2]))
					outfile.write('C' + ' ' + img1 + ' ' + img2 + ' ' + '1' + ' ' + cat_pair + '\n')

		for i in related['similar']:

			# Add 1 positive compatibility example
			img2 = url_map[i]
			cat2 = category_map[i]
			cat_pair = ''.join(sorted([cat1, cat2]))
			outfile.write('C' + ' ' + img1 + ' ' + img2 + ' ' + '0' + ' ' + cat_pair + '\n')

			# Find 10 negative compatibility examples
			for c in inverse_category_maps.keys():
				if c!=cat1:
					continue
				for j in choice(len(inverse_category_maps[c]), 5):

					asin2 = category_map[j]
					if asin2 in related['similar']:
						continue

					cat2 = category_map[asin2]
					img2 = url_map[asin2]
					cat_pair = ''.join(sorted([cat1, cat2]))
					outfile.write('C' + ' ' + img1 + ' ' + img2 + ' ' + '1' + ' ' + cat_pair + '\n')

		if len(related['similar'])==0:
			for c in inverse_category_maps.keys():
				if c!=cat1:
					continue
				for j in choice(len(inverse_category_maps[c]), 5):

					asin2 = category_map[j]
					if asin2 in related['similar']:
						continue

					cat2 = category_map[asin2]
					img2 = url_map[asin2]
					cat_pair = ''.join(sorted([cat1, cat2]))
					outfile.write('C' + ' ' + img1 + ' ' + img2 + ' ' + '1' + ' ' + cat_pair + '\n')

		count += 1
		if count==0:
			break






