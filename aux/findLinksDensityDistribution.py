import json
from trie import ManageTrie

class FindLinksDensityDistribution:

	def __init__(self, jsonFile):
		self.f = open(jsonFile)
		self.computeDensity()
		self.tr = ManageTrie()
		self.compatibility_mapFrom = {}
		self.compatibility_mapTo = {}
		self.similarity_mapTo = {}
		self.similarity_mapFrom = {}

	def computeDensity(self):
		
		for line in self.f:
			info = json.loads(line.rstrip())
			anchor_asin = info["asin"]

			for asins in info["related"]["compatible"]:
				
	


FindLinksDensityDistribution("/data/srajpal2/AmazonDataset/GoldStandard/training_images.json")
