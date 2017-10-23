from datetime import datetime

pathToMetaFile = "/data/srajpal2/deducplicatedMeta.json"
pathToDeduplicatedFile = "/data/srajpal2/deduplicatedCategoriesMeta.json"

outfile = open(pathToDeduplicatedFile, 'w')

with open(pathToMetaFile) as f:
	count = 0
	for line in f:
		count += 1
                if count%100000==0:
                        print count/(1.0*1503384), datetime.now()
		categories_loc = line.find("'categories': ")
		if categories_loc==-1:
			continue
		outfile.write(line)
