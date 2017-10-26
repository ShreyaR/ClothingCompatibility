from random import uniform

metafile = "/data/srajpal2/deduplicatedMeta.json"
lineCount = 1200081.0

trainFile = open("/data/srajpal2/AmazonDataset/trainImgs.txt", "w")
testFile = open("/data/srajpal2/AmazonDataset/testImgs.txt", "w")
valFile = open("/data/srajpal2/AmazonDataset/valImgs.txt", "w")

with open(metafile) as f:
	count = 0
	for line in f:
		asin_beginning = line.find("'asin': ") + 9
		asin_end = line.find("'", asin_beginning)
		asin = line[asin_beginning:asin_end]
		assignment = uniform(0,1)
		if 0<=assignment<0.8:
			trainFile.write(asin+'\n')
		elif 0.8<=assignment<0.99:
			testFile.write(asin+'\n')
		else:
			valFile.write(asin+'\n')
		count += 1
		if count%10000==0:
			print "%.2f" % (100*count/lineCount)
trainFile.close()
testFile.close()
valFile.close()
