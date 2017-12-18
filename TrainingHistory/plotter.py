import matplotlib.pyplot as plt
import numpy as np

class Plotter:
	def __init__(self, list_of_versions):
		self.list_of_versions = list_of_versions
		trainingHist = {i:self.readTrainingHistory(i) for i in self.list_of_versions}
		valHistory = {i:self.readValidationHistory(i) for i in self.list_of_versions}

	def readTrainingHistory(self, version):

	def readValidationHistory(self, version):

	def plotTrainingLossOverIterations(self)

