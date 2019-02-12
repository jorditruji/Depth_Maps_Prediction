from Models.pspnet import PSPNet
from Data_management.dataset import Dataset
import torch
from torch.utils import data
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn


class NNTrainer(nn.Module):
    def __init__(self, session_name, network):
    	self.session_name = session_name+'_'+str(datetime.today())
    	self.net = network

        
    