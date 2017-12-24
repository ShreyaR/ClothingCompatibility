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
			print info["related"], type(info["related"])
			break

FindLinksDensityDistribution()
