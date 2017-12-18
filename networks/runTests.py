import os

lr = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
versions = range(37,37+len(lr))

for learning_rate, version_num in zip(lr, versions):
	
	print "\nStarting version %s\n############################\n" % (version_num)
	os.system("python similarity_training.py %d %f 5000 SGD 1 2" % (version_num, learning_rate))




