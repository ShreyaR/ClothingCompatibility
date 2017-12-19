import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

class LearningRatePlot:
	"""
	Plots Best Learning Rates over a range for all 4 algorithms
	"""

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
			training_history_smoothing=10, val_history_smoothing=0,
			train_val_same_plot=True, all_versions_same_plot=True):
		
		num_opts = len(opts)
		fig = plt.figure()
	
		lines = {}
		count = 1
		for o in opts:
			ax = fig.add_subplot(2,2,count)
			count+=1

			training_history = np.array(training_history_dict[o])
			train_moving_avg_loss = self.running_mean(training_history[:,1], training_history_smoothing)
			lines["%s: Train" % (o)] ,= ax.plot(training_history[training_history_smoothing:,0], train_moving_avg_loss, label="%s: Train" % (o))

			val_history = np.array(val_history_dict[o])
			val_moving_avg_loss = self.running_mean(val_history[:,1], val_history_smoothing)
			lines["%s: Test" % (o)] ,= ax.plot(val_history[val_history_smoothing:,0], val_moving_avg_loss, 
					label="%s: Test" % (o))#, color=lines["%s: Train" % (o)].get_color())			

			ax.legend(handles=lines.values())
			lines = {}
		# ax.set_xscale('log')
		plt.suptitle('Training and Validation History for various Optimization Algorithms')
		plt.show()


	def running_mean(self, x, N):
		# cumsum = np.cumsum(np.insert(x, 0, 0))
		if N==0:
			return x

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
		GeneralPurposePlotter(training_hist_dict.keys(), training_hist_dict, val_history_dict, training_history_smoothing=50)

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


class TrainValDiffPlot:

	def __init__(self, opts, t, num_samples):

		self.num_samples = num_samples

		train, val = self.get_train_val_matrix(opts)

		mean1, var1 = np.mean(train,axis=1), np.var(train, axis=1)
		mean2, var2 = np.mean(val,axis=1), np.var(val, axis=1)

		mean_array = mean1 - mean2
		sigma_array = np.sqrt((var1 + var2)/self.num_samples)

		lower, upper = mean_array-(t*sigma_array), mean_array+(t*sigma_array)
		self.plot_confidence_bounds(mean_array, lower, upper, mean1, mean2, 100)


	def get_train_val_matrix(self, optim):

		val = [[] for _ in xrange(self.num_samples)]
		train = [[] for _ in xrange(self.num_samples)]

		for i in xrange(self.num_samples):
			with open("RandomTrainTestGaps/%s/val_training_loss_V%d.txt" % (optim,i+1)) as f:
				for line in f:
					line_info = line.rstrip().split(', ')
					val[i].append(float(line_info[2]))
					train[i].append(float(line_info[1]))

		val = np.array(val).T
		train = np.array(train).T

		return train, val

	def plot_confidence_bounds(self, mean, lower, upper, mean_train, mean_val, moving_average):

		fig = plt.figure()
		lines = {}
		x_ticks = np.arange(mean.size)

		# subsampled_pts = range(0, 2001, 50)
		# Left subplot for comparison
		ax = fig.add_subplot(1,2,1)
		lines["Train"] ,= ax.plot(x_ticks[moving_average:], self.running_mean(mean_train, moving_average), label="Mean Training Loss")
		lines["Test"] ,= ax.plot(x_ticks[moving_average:], self.running_mean(mean_val, moving_average), label="Mean Testing Loss")
		ax.legend(handles=lines.values())
		lines = {}

		ax = fig.add_subplot(1,2,2)
		lines["Mean"] ,= ax.plot(x_ticks[moving_average:], self.running_mean(mean, moving_average), label="Testing Gap: Mean")
		lines["Lower"] ,= ax.plot(x_ticks[moving_average:], self.running_mean(lower, moving_average), label="Testing Gap: Lower")
		lines["Upper"] ,= ax.plot(x_ticks[moving_average:], self.running_mean(upper, moving_average), label="Testing Gap: Upper", color=lines["Lower"].get_color())
		ax.fill_between(x_ticks[moving_average:], self.running_mean(lower, moving_average), self.running_mean(upper, moving_average), color=lines["Lower"].get_color(), alpha=0.5)
		ax.legend(handles=lines.values())
		plt.show()
		return

	def running_mean(self, x, N):
		if N==0:
			return x

		cumsum = np.cumsum(x)
		return (cumsum[N:] - cumsum[:-N]) / float(N)




class GapDiffPlot:

	def __init__(self, opt1, opt2, t, num_samples):

		self.num_samples = num_samples

		gaps1 = self.get_gap_matrix(opt1)
		gaps2 = self.get_gap_matrix(opt1)

		mean1, var1 = np.mean(gaps1,axis=1), np.var(gaps1, axis=1)
		mean2, var2 = np.mean(gaps2,axis=1), np.var(gaps2, axis=1)

		mean_array = mean1 - mean2
		sigma_array = np.sqrt((var1 + var2)/self.num_samples)

		lower, upper = mean_array-(t*sigma_array), mean_array+(t*sigma_array)
		self.plot_confidence_bounds(mean_array, lower, upper, mean1, mean2, 50)


	def get_gap_matrix(self, optim):

		gap = [[] for _ in xrange(self.num_samples)]

		for i in xrange(self.num_samples):
			with open("RandomTrainTestGaps/%s/val_training_loss_V%d.txt" % (optim,i+1)) as f:
				for line in f:
					line_info = line.rstrip().split(', ')
					gap[i].append(float(line_info[2])-float(line_info[1]))
					
		gap = np.array(gap).T
		return gap

	
	def plot_confidence_bounds(self, mean, lower, upper, mean_train, mean_val, moving_average):

		fig = plt.figure()
		lines = {}
		x_ticks = np.arange(mean.size)

		# subsampled_pts = range(0, 2001, 50)
		
		# Left subplot for comparison
		ax = fig.add_subplot(1,2,1)
		lines["Gaps1"] ,= ax.plot(x_ticks[moving_average:], self.running_mean(mean_train, moving_average), label="Testing Gaps 2")
		lines["Gaps2"] ,= ax.plot(x_ticks[moving_average:], self.running_mean(mean_val, moving_average), label="Testing Gaps 2")
		ax.legend(handles=lines.values())
		lines = {}

		ax = fig.add_subplot(1,2,2)
		lines["Mean"] ,= ax.plot(x_ticks[moving_average:], self.running_mean(mean, moving_average), label="Testing Gap: Mean")
		lines["Lower"] ,= ax.plot(x_ticks[moving_average:], self.running_mean(lower, moving_average), label="Testing Gap: Lower")
		lines["Upper"] ,= ax.plot(x_ticks[moving_average:], self.running_mean(upper, moving_average), label="Testing Gap: Upper")
		ax.legend(handles=lines.values())
		plt.show()
		return

	def running_mean(self, x, N):
		if N==0:
			return x

		cumsum = np.cumsum(x)
		return (cumsum[N:] - cumsum[:-N]) / float(N)




TrainValDiffPlot("SGD", 2.306, 5)




# LearningRatePlot(range(16, 16+28))
# TrainingValHistory([18,25,33,42])
