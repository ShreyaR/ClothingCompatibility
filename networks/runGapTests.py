import os

#versions = range(37,37+len(lr))
versions = [3,4]
opt = ['SGD']
learning_rate = [0.1]

for optim,lr in zip(opt, learning_rate):
	for version_num in versions:
	
		print "\nStarting %s version %d\n############################\n" % (optim, version_num)
		os.system("python similarity_training_and_val.py %d %f 2000 %s 1" % (version_num, lr, optim))


#versions = [10, 1, 10]
#opt = ['SGD', 'Adagrad', 'Adagrad']
#learning_rate = [0.1, 0.001, 0.001]

#for version_num, optim, lr in zip(versions, opt, learning_rate):
#	print "\nStarting %s version %d\n############################\n" % (optim, version_num)
#	os.system("python similarity_training_and_val.py %d %f 2000 %s 1" % (version_num, lr, optim))
	

