import json
from trie import ManageTrie

infile = open("/data/srajpal2/AmazonDataset/meta_tbs.json")
outfile = open("/data/srajpal2/AmazonDataset/updated_categories_meta_tbs.json", "a")

asin_lookup = ManageTrie()
category_map = {}


for line in infile:
	info = json.loads(line.rstrip())
	asin_lookup.add_to_trie(info["asin"])
	category_map[info["asin"]] = info["category"]

for line in infile:
	info = json.loads(line.rstrip())
	related = info["related"]
	own_cat = info["category"]

	new_similar = []
	new_compatible = []

	if 'similar' in related.keys():
		for asin in related['similar']:
			if asin_lookup.lookup_in_trie(asin):
				print cat, own_cat
				cat = category_map[asin]
				if cat==own_cat:
					new_similar.append(asin)
				else:
					new_compatible.append(asin)
	if 'compatible' in related.keys():
		for asin in related['compatible']:
			if asin_lookup.lookup_in_trie(asin):
				cat = category_map[asin]
				if cat==own_cat:
					new_similar.append(asin)
				else:
					new_compatible.append(asin)

	break

	related = {"similar": new_similar, "compatible": new_compatible}

	json.dump({"asin": info['asin'], "category":own_cat, "imUrl":info["imUrl"], "related":related}, outfile)
	outfile.write('\n')