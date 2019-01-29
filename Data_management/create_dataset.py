from glob import glob
from random import shuffle
import numpy as np


'One use script to create the dataset partitions and store them in dataset.npy'



def read_scenes(path):
	# Reads all scenes (folders) of the dataset
	return glob(path+'*/')


def get_depths_from_folder(path):
	# Read all depth captures of a folder
	return glob(path+'*.pgm')


def make_partitions(scenes,n_partitions=1):
	#Partitions train val test 0.6 0.2 0.2 
	n_scenes = len(scenes)
	# Initialize dict to store data paths
	dataset = {}
	dataset['train']=[]
	dataset['val']=[]
	dataset['test']=[]
	print "Found {}  scenes".format(n_scenes)
	for _i in range(n_partitions):
		shuffle(scenes)
		train = scenes[0:int(0.6*n_scenes)]
		val_test = scenes[int(0.6*n_scenes):]
		val =  val_test[0:int(0.5*len(val_test))]
		test =  val_test[int(0.5*len(val_test)):]
		print "{} training scenes\n{} validation scenes\n{} testing scenes".format(len(train), len(val), len(test))
		
		# Read depth file names and add them to dataset splits
		for _j, scene in enumerate(train):
			dataset['train']+=get_depths_from_folder(scene)
		print "Train samples {}".format(len(dataset['train']))

		for _j, scene in enumerate(val):
			dataset['val']+=get_depths_from_folder(scene)
		print "Validation samples {}".format(len(dataset['val']))

		for _j, scene in enumerate(test):
			dataset['test']+=get_depths_from_folder(scene)
		print "Test samples {}".format(len(dataset['test']))

		print "Saving data!!!! Finished"
		np.save('dataset',dataset)

if __name__ == '__main__':
	DATA_PATH = '/projects/world3d/2017-06-scannet/'
	scenes = read_scenes(DATA_PATH)
	make_partitions(scenes)
