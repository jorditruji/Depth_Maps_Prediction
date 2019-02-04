from Models.pspnet import PSPNet
from Data_management.dataset import Dataset
import torch
from torch.utils import data
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

# Create model
models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
    }
    # Instantiate a model

net = models['resnet18']()
depths = ['Test_samples/frame-000000.depth.pgm','Test_samples/frame-000025.depth.pgm','Test_samples/frame-000050.depth.pgm','Test_samples/frame-000075.depth.pgm']
dataset = Dataset(depths)

# dataset = Dataset(np.load('Data_management/dataset.npy').item()['train'][1:20])
# Parameters
params = {'batch_size': 4,
          'shuffle': True,
        'num_workers': 8}
training_generator = data.DataLoader(dataset,**params)
net.train()
print(net)

optimizer_ft = optim.Adagrad(net.parameters(), lr=0.001, lr_decay=0)

for a in range(150):
    for depths, rgbs in training_generator:
        print(depths.size(), rgbs.size())
        rgbs = Variable(rgbs)
        


