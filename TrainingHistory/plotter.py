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


class GeneralPurposePlotter:

	def __init__(self, opts, training_history_dict, val_history_dict=None,
			training_history_smoothing=10, val_history_smoothing=1,
			train_val_same_plot=True, all_versions_same_plot=True):
		num_opts = len(opts)

		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)

		lines = {}
		for o in opts:
			training_history = np.array(training_history_dict[o])
			moving_avg_loss = self.running_mean(training_history[:,1], training_history_smoothing)
			# print moving_avg_loss.shape, training_history[10:,:].shape
			lines["%s: Train" % (o)] ,= ax.plot(training_history[training_history_smoothing:,0], moving_avg_loss, label="%s: Train" % (o))

		plt.legend(handles=lines.values())
		# ax.set_xscale('log')
		plt.show()


	def running_mean(self, x, N):
		# cumsum = np.cumsum(np.insert(x, 0, 0))
		cumsum = np.cumsum(x)
		return (cumsum[N:] - cumsum[:-N]) / float(N)





class TrainingValHistory:


	def __init__(self, versions):
		training_hist_dict = {}
		val_history_dict = {}
		for v in versions:
			info = self.getInfo(v)
			training_hist = self.readTrainingHist(v)
			val_hist = self.readValHist(v)
			training_hist_dict[info] = training_hist
			val_history_dict[info] = val_hist
		GeneralPurposePlotter(training_hist_dict.keys(), training_hist_dict, val_history_dict, training_history_smoothing=100)

	def getInfo(self, v):
		with open("V%d/Similarity/info.txt" % v) as f:
			info = [x.rstrip().split(': ')[1] for x in f.readlines()]
		return info[3]

	def readTrainingHist(self,v):
		info = []
		with open("V%d/Similarity/training_loss.txt" % v) as f:
			for line in f:
				iteratn, loss = line.rstrip().split(', ')[:2]
				info.append([int(iteratn), float(loss)])
		return info

	def readValHist(self,v):
		info = []
		with open("V%d/Similarity/val_loss.txt" % v) as f:
			for line in f:
				iteratn, loss = line.rstrip().split(', ')[:2]
				info.append([int(iteratn), float(loss)])
		return info


# LearningRatePlot(range(16, 16+28))
TrainingValHistory([16,23])
