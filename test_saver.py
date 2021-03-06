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
import torch.nn.functional as F
import matplotlib 
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
from matplotlib import path, rcParams
import matplotlib.pyplot as plt


# Save predictions

def save_predictions(name,prediction, rgb, depth):
    # Matplotlib style display = channels last
    inp = rgb.numpy().transpose((1, 2, 0))
    mean = np.array([0.4944742,  0.4425867,  0.38153833])
    std = np.array([0.23055981, 0.22284868, 0.21425385])
    # inp = std * inp + mean
    print(inp.shape)
    plt.subplot(3,1,1)
    plt.imshow(inp)
    plt.title("RGB")
    print(depth.shape)
    #Depth
    plt.subplot(3,1,2)
    plt.imshow(np.squeeze(depth.numpy()), 'gray', interpolation='nearest')
    plt.title("Ground truth")

    plt.subplot(3,1,3)
    plt.imshow(np.squeeze(prediction.numpy()), 'gray', interpolation='nearest')
    plt.title("Prediction")

    plt.show()
    plt.savefig(name+'.png')
    return





if __name__ == '__main__':
    # Testing:
    # Sample data


    depths = np.load('Data_management/dataset.npy').item()['train']
    dataset = Dataset(depths)
    sample_depth, sample_RGB, path = dataset.__getitem__(0)
    #print(sample_depth, sample_RGB)
    print(sample_RGB.size())#, sample_depth.size())
    save_predictions('', sample_RGB,sample_depth)
