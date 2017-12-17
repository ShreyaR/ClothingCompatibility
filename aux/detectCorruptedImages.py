"""Detects those images which exist in the data directory but are corrupted"""
from PIL import Image
import json

def findCorruptedImgs(jsonFile):
	outfile = open("corruptedImages.txt","a")
	with open(jsonFile) as f:
		for line in f:
			info = json.loads(line.rstrip())
			url = info["imUrl"]
			try:
				Image.open(url)
			except IOError:
				outfile.write(info["asin"]+"\n")
	outfile.close()
	return

findCorruptedImgs("/data/srajpal2/AmazonDataset/training_images.json")
	
