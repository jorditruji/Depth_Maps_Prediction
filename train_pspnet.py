from Models.pspnet import PSPNet
from Data_management.dataset import Dataset
import torch
from torch.utils import data
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm

#LR decay:
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



#Losses:
class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        print("Calculing loss")
        print( torch.abs(torch.log(real)-torch.log(fake)) )
        loss = torch.sqrt( torch.mean(torch.log(real)-torch.log(fake) ** 2 ) )
        return loss


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        loss = torch.sqrt( torch.mean( torch.abs(10.*real-10.*fake) ** 2 ) )
        return loss


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    
    # L1 norm
    def forward(self, grad_fake, grad_real):
        
        return torch.sum( torch.mean( torch.abs(grad_real-grad_fake) ) )


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

# Instantiate a model and dataset
net = models['resnet34']()
depths = np.load('Data_management/dataset.npy').item()
#depths = ['Test_samples/frame-000000.depth.pgm','Test_samples/frame-000025.depth.pgm','Test_samples/frame-000050.depth.pgm','Test_samples/frame-000075.depth.pgm']
dataset = Dataset(depths['train'])
dataset_val = Dataset(depths['val'])

# dataset = Dataset(np.load('Data_management/dataset.npy').item()['train'][1:20])
# Parameters
params = {'batch_size': 6 ,
          'shuffle': True,
          'num_workers': 4}

training_generator = data.DataLoader(dataset,**params)
val_generator = data.DataLoader(dataset_val,**params)

net.train()
print(net)


class CartPoleConfig:
    learning_rate = 0.001

# Loss
depth_criterion = RMSE()

# Use gpu if possible and load model there
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = net.to(device)

# Optimizer
optimizer_ft = optim.Adagrad(net.parameters(), lr=2e-4, lr_decay=0)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
loss = []
for epoch in range(30):
    # Train
    net.train()
    cont = 0
    for depths, rgbs in training_generator:
        cont+=1
        # Get items from generator
        outputs = depths.to(device)
        inputs = rgbs.to(device)

        # Clean grads
        optimizer_ft.zero_grad()

        #Forward
        predict_depth = net(inputs)

        #Sobel grad estimates:
        predict_grad = net.imgrad(predict_depth)
        real_grad = net.imgrad(outputs)

        #Backward+update weights
        depth_loss = depth_criterion(predict_depth, outputs)+depth_criterion(predict_grad, real_grad)
        depth_loss.backward()
        optimizer_ft.step()
        loss.append(depth_loss.item())
        sys.stdout.write('\r%s %s %s %s %s %s' % ('Processing training batch: ', cont, '/', training_generator.__len__(),' with loss: ', depth_loss.item())),
        sys.stdout.flush()


    print("[epoch %2d] loss: %.4f " % (epoch, depth_loss ))
    
    # Val
    net.eval()
    loss_val = []
    cont = 0

    # We dont need to track gradients here, so let's save some memory and time
    with torch.no_grad():
        for depths, rgbs in val_generator:
            cont+=1
            # Get items from generator
            outputs = depths.to(device)
            inputs = rgbs.to(device)


            #Forward
            predict_depth = net(inputs)

            #Sobel grad estimates:
            predict_grad = net.imgrad(predict_depth)
            real_grad = net.imgrad(outputs)

            depth_loss = depth_criterion(predict_depth, outputs)+depth_criterion(predict_grad, real_grad)
            
            sys.stdout.write('\r%s %s %s %s %s %s ' % ('Processing training batch: ', cont, '/', val_generator.__len__(),' with loss: ', depth_loss.item())),
            sys.stdout.flush()       
            
            loss_val.append(depth_loss.item())
            #scheduler.step()
        if epoch%2==0:
            predict_depth = predict_depth.detach().cpu()
            np.save('v3_pred'+str(epoch), predict_depth)


        print("[epoch %2d] val loss: %.4f " % (epoch, depth_loss ))

predict_depth = predict_depth.cpu()
np.save('first_pred', predict_depth)

np.save('loss',loss)
np.save('loss_val',loss_val)
