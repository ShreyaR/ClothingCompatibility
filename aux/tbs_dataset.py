import json
import ast

tops = {'Button-Down & Dress Shirts', 'Fashion Hoodies & Sweatshirts', 'Shirts', 'Sweaters', 'Tops & Tees'}
bottoms = {'Jeans', 'Leggings', 'Pants', 'Pants & Capris', 'Shorts', 'Skirts', 'Skirts, Scooters & Skorts'}
shoes = {'Shoes'}

outfile = open('/data/srajpal2/AmazonDataset/meta_tbs.json', 'a')

with open('/data/srajpal2/AmazonDataset/meta_Clothing_Shoes_and_Jewelry.json') as f:
	count = 0
	for line in f:

                if count==1000:
                        count
                count += 1

		info = ast.literal_eval(line.rstrip())

		try:	
			category_vec = info['categories']
		except KeyError:
			continue

		category = None
		for i in category_vec:
			for j in i:
				if j in tops:
					category='t'
					break
				elif j in bottoms:
					category='b'
					break
				elif j in shoes:
					category='s'
					break
		if category==None:
			continue

		try:
			imUrl = '/data/srajpal2/AmazonData/image_downloads/' + info['imUrl'].split('/')[-1]
		except KeyError:
			continue

		try:
			related = info['related']
			similar = []
			compatible = []
			if 'also_viewed' in related.keys():
				similar += related['also_viewed']
			if 'buy_after_viewing' in related.keys():
				similar += related['buy_after_viewing']
			if 'also_bought' in related.keys():
				compatible += related['also_bought']
			if 'bought_together' in related.keys():
				compatible += related['bought_together']
			related = {'similar':similar, 'compatible':compatible}
		except KeyError:
			related = {}

		
		json.dump({"asin": info['asin'], "category":category, "imUrl":imUrl, "related":related}, outfile)
		outfile.write('\n')
	
		

