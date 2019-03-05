from Models.pspnet import PSPNet
from Data_management.dataset import Dataset
import torch
from torch.utils import data
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import sys
import copy
import pickle


depths = np.load('Data_management/dataset.npy').item()
#depths = ['Test_samples/frame-000000.depth.pgm','Test_samples/frame-000025.depth.pgm','Test_samples/frame-000050.depth.pgm','Test_samples/frame-000075.depth.pgm']
dataset = Dataset(depths['train'])
dataset_val = Dataset(depths['val'])

# dataset = Dataset(np.load('Data_management/dataset.npy').item()['train'][1:20])
# Parameters
params = {'batch_size': 1 ,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True}

training_generator = data.DataLoader(dataset,**params)
val_generator = data.DataLoader(dataset_val,**params)


for depth, rgb, path in training_generator:
	print(path[0])
	np_depth = depth.numpy()
	new_name = path[0].replace('.depth','_np.depth',1)
	print(new_name)
	with open(new_name, "wb") as pickle_out:
		pickle.dump(randomlist, pickle_out)


