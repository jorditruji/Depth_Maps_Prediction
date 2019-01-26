from glob import glob
from random import shuffle

def read_scenes(path):
	# Reads all scenes (folders) of the dataset
	return glob(path+'*/')


def get_depths_from_folder(path):
	# Read all depth captures of a folder
	return glob(path+'*.pgm')


def make_partitions(scenes,n_partitions):
	#Partitions 0.6 0.2 0.2 
	n_scenes = len(scenes)
	for _ i in range(n_partitions):
		shuffled_scenes = shuffle(scenes)
		train = scenes[0:int(0.6*n_scenes)]
		val_test = scenes[int(0.6*n_scenes):]
		val =  val_test[0:int(0.5*len(val_test))]
		test =  val_test[int(0.5*len(val_test)):]




DATA_PATH = '/projects/world3d/2017-06-scannet'