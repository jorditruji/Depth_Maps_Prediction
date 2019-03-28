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
        loss = torch.sqrt( torch.mean( (10.*real-10.*fake) ** 2 ) )
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
net = models['resnet18']()
depths = np.load('Data_management/dataset.npy').item()
#depths = ['Test_samples/frame-000000.depth.pgm','Test_samples/frame-000025.depth.pgm','Test_samples/frame-000050.depth.pgm','Test_samples/frame-000075.depth.pgm']
dataset = Dataset(depths['train'])
dataset_val = Dataset(depths['val'])

# dataset = Dataset(np.load('Data_management/dataset.npy').item()['train'][1:20])
# Parameters
params = {'batch_size': 24 ,
          'shuffle': True,
          'num_workers': 16,
          'pin_memory': True}

training_generator = data.DataLoader(dataset,**params)
val_generator = data.DataLoader(dataset_val,**params)

net.train()
print(net)


# Loss
depth_criterion = RMSE()

# Use gpu if possible and load model there
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = net.to(device)

# Optimizer
optimizer_ft = optim.Adagrad(net.parameters(), lr=2e-4, lr_decay=0)
#scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
loss = []
history_val = []
best_loss = 50
for epoch in range(30):
    # Train
    net.train()
    cont = 0
    loss_train = 0.0
    for depths, rgbs, filename in training_generator:
        cont+=1
        # Get items from generator
        inputs = rgbs.cuda()

        # We wont use depths until the RGB is forwarded,
        # so it can be parallelized with computation
        outputs = depths.cuda(async=True)
        # Clean grads
        optimizer_ft.zero_grad()

        #Forward
        predict_depth, predict_grad = net(inputs)
        
        #Sobel grad estimates:
        real_grad = net.imgrad(outputs)

        #Backward+update weights
        depth_loss = depth_criterion(predict_depth, outputs)+depth_criterion(predict_grad, real_grad)
        depth_loss.backward()
        optimizer_ft.step()
        loss_train+=depth_loss.item()*inputs.size(0)
        save_predictions(name,predict_depth[0], rgbs[0], outputs[0])
        if cont%250 == 0:
            #loss.append(depth_loss.item())
            print("TRAIN: [epoch %2d][iter %4d] loss: %.4f" \
            % (epoch, cont, depth_loss.item()))
    loss_train = loss_train/dataset.__len__()
    print("\n FINISHED TRAIN epoch %2d with loss: %.4f " % (epoch, loss_train ))
    # Val
    loss.append(loss_train)
    net.eval()
    loss_val = 0.0
    cont = 0

    # We dont need to track gradients here, so let's save some memory and time
    with torch.no_grad():
        for depths, rgbs, filename in val_generator:
            cont+=1
            # Get items from generator
            inputs = rgbs.cuda()
            # Non blocking so computation can start while labels are being loaded
            outputs = depths.cuda(async=True)
            
            #Forward
            predict_depth , predict_grad= net(inputs)

            #Sobel grad estimates:
            real_grad = net.imgrad(outputs)

            depth_loss = depth_criterion(predict_depth, outputs)+depth_criterion(predict_grad, real_grad)
            loss_val+=depth_loss.item()*inputs.size(0)
            if cont%250 == 0:
                print("VAL: [epoch %2d][iter %4d] loss: %.4f" \
                % (epoch, cont, depth_loss))   
            
            #scheduler.step()
        if epoch%2==0:
            predict_depth = predict_depth.detach().cpu()
            saver['names'] = filename
            saver['img'] = predict_depth
            np.save('pspnet'+str(epoch), saver)

        loss_val = loss_val/dataset_val.__len__()
        history_val.append(loss_val)
        print("\n FINISHED VAL epoch %2d with loss: %.4f " % (epoch, loss_val ))
        
    if loss_val< best_loss and epoch>2:
        best_loss = depth_loss
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(net.state_dict(), 'model_pspnet')


predict_depth = predict_depth.cpu()
np.save('first_pred', predict_depth)

np.save('loss_psp',loss)
np.save('loss_val_psp',history_val)
