# -*- coding: utf-8 -*-
from __future__ import division
from dataset import Dataset
import numpy as np
from PIL import Image
from depth_preprocessing import read_depth
import sys

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=2, bar_length=40):
	"""
	Call in a loop to create terminal progress bar
	@params:
	iteration   - Required  : current iteration (Int)
	total       - Required  : total iterations (Int)
	prefix      - Optional  : prefix string (Str)
	suffix      - Optional  : suffix string (Str)
	decimals    - Optional  : positive number of decimals in percent complete (Int)
	bar_length  - Optional  : character length of bar (Int)
	"""
	str_format = "{0:." + str(decimals) + "f}"
	percents = str_format.format(100 * (iteration / float(total)))
	filled_length = int(round(bar_length * iteration / float(total)))
	bar = '█' * filled_length + '-' * (bar_length - filled_length)

	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()

train_dict = np.load('dataset.npy').item()['train']
test =['../Test_samples/frame-000000.depth.pgm','../Test_samples/frame-000025.depth.pgm','../Test_samples/frame-000050.depth.pgm','../Test_samples/frame-000075.depth.pgm']
    
means = []
stds = []
max_depth = []
min_depth = []
dataset = Dataset(train_dict)
total = len(train_dict)
cont = 1
for sample,depth in zip(dataset.RGB_frames,dataset.depth_frames):
	print_progress(cont, total, prefix='Progress:', suffix='Complete', bar_length=50)
	cont+=1
	with open(sample, 'rb') as f:
		img = Image.open(f).convert('RGB')
	img = np.asarray(img)/255
	width_ ,height, n_channels = img.shape
	val = img.reshape(width_*height,n_channels)
	means.append(np.mean(val,axis=0))
	stds.append(np.std(val,axis=0))
	depth_img = read_depth(depth)
	try:
		max_depth.append(np.max(depth_img))
		min_depth.append(np.amin(depth_img[depth_img>0]))
	except:
		print("failes")

means = np.array(means)
stds = np.array(stds)

global_mean = np.mean(means,axis = 0)
global_std = np.mean(stds,axis = 0)

max_save = np.max(max_depth)


min_save = np.min(min_depth)

print "Max:{}".format(max_save)
print "Min: {}".format(min_save)
print "Mean: {}".format(global_mean)
print "Std: {}".format(global_std)

np.save ('means',global_mean)
np.save ('stds',global_std)
np.save ('max_save',max_save)
np.save ('min_save',min_save)