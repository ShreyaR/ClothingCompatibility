import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

class LearningRatePlot:

	def __init__(self, list_of_versions):
		self.list_of_versions = list_of_versions
		self.train_data = {x:[] for x in ['Adam', 'Adagrad', 'RMSprop', 'SGD']}
		#trainingHist = {i:self.readTrainingHistory(i) for i in self.list_of_versions}
		#:valHistory = {i:self.readValidationHistory(i) for i in self.list_of_versions}
		for v in self.list_of_versions:
			self.readTrainingHistory(v, final_loss_avg=50)

		for k,v in self.train_data.items():
			self.train_data[k] = sorted(v,key=lambda l:l[0])

		pprint(self.train_data)
		self.plotLearningRate()

	def readTrainingHistory(self, version, final_loss_avg=10):
		with open("V%d/Similarity/info.txt" % (version)) as f:
			info = [x.rstrip() for x in f.readlines()]
		learning_rate = float(info[0].split(': ')[1])
		opt = info[3].split(': ')[1]

		with open("V%d/Similarity/training_loss.txt" % (version)) as f:
			training_loss = f.readlines()
		if len(training_loss)>final_loss_avg:
			training_loss = training_loss[-10:]

		final_training_loss = [float(x.rstrip().split(', ')[1]) for x in training_loss]
		final_training_loss_avg = sum(final_training_loss)/len(final_training_loss)
		self.train_data[opt].append([learning_rate, final_training_loss_avg])
		return


	#def readValidationHistory(self, version):

	def plotLearningRate(self):

		lines = {}

		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)

		for k,v in self.train_data.items():
			v = np.array(v)
			lines[k] ,= ax.plot(v[:,0], v[:,1], label=k)

		plt.legend(handles=lines.values())
		ax.set_xscale('log')
		plt.show()

		

LearningRatePlot(range(16, 16+28))
