# -*- coding: utf-8 -*-
from dataset import Dataset
import numpy as np
from PIL import Image
from depth_preprocessing import read_depth


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

means = []
stds = []
max_depth = []
min_depth = []
dataset = Dataset(train_dict)
printProgressBar(count_progress+1, len(labels), prefix='Progress:', suffix='Complete', length=50)
total = len(train_dict)
cont = 1
for sample,depth in zip(dataset.RGB_frames,dataset.depth_frames):
	printProgressBar(count_progress+1, total, prefix='Progress:', suffix='Complete', length=50)
	cont+=1
	with open(sample, 'rb') as f:
		img = Image.open(f).convert('RGB')
	img = np.asarray(img)
	width_ ,height, n_channels = img.shape
	val = img.reshape(width_*height,n_channels)
	means.append(np.mean(val,axis=0))
	stds.append(np.std(val,axis=0))
	depth_img = read_depth(depth)
	max_depth.append( np.max(depth_img))
	min_depth.append(np.min(depth_img[depth_img>0]))

means = np.array(means)
stds = np.array(stds)

global_mean = np.mean(means,axis = 0)
global_std = np.mean(stds,axis = 0)

max_save = np.max(max_depth)
min_save = np.min(min_depth)


np.save ('means',global_mean)
np.save ('stds',global_std)
np.save ('max_save',max_save)
np.save ('means',min_save)