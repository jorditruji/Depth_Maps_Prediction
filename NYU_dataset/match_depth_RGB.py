import glob
import numpy as np


def read_folder_names(input_path):
	return glob.glob(self.input_path+'*')

def read_depth_frames(input_path):
	return glob.glob(self.input_path+'/*.pgm')

def get_time_stamp(filename):
	tstamp = 

def main(dataset_path):
	pair_list = []
	scenes = read_folder_names(dataset_path)
	for scene in scenes:
		depth_names = read_depth_frames(scene)


