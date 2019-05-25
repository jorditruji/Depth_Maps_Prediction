from Models.extractors import *
from Data_management.dataset import Dataset
import torch
from torch.utils import data
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import sys
import copy
from Models.unet_v3 import ResNetUNet_V2
import torch.nn.functional as F

import matplotlib 
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
from matplotlib import path, rcParams
import matplotlib.pyplot as plt


# Save predictions

def save_predictions(prediction, rgb, depth, name = 'test'):
    # Matplotlib style display = channels last
    inp = rgb.numpy().transpose((1, 2, 0))
    mean = np.array([0.4944742,  0.4425867,  0.38153833])
    std = np.array([0.23055981, 0.22284868, 0.21425385])
    # inp = std * inp + mean
    plt.subplot(3,1,1)
    plt.imshow(inp)
    plt.title("RGB")
    #Depth
    plt.subplot(3,1,2)
    plt.imshow(np.squeeze(depth.cpu().numpy()), 'gray', interpolation='nearest')
    plt.title("Ground truth")

    plt.subplot(3,1,3)
    plt.imshow(np.squeeze(prediction.cpu().numpy()), 'gray', interpolation='nearest')
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
        #print("Calculing loss")
        #print(torch.min(fake), torch.min(real))
        #print(torch.max(fake), torch.max(real))
        loss = torch.sqrt( torch.mean(torch.abs(torch.log(real+1e-3)-torch.log(fake+1e-3)) ** 2 ) )
        #print(loss)
        return loss


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
    
    def forward(self, grad_fake, grad_real):
        prod = ( grad_fake[:,:,None,:] @ grad_real[:,:,:,None] ).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt( torch.sum( grad_fake**2, dim=-1 ) )
        real_norm = torch.sqrt( torch.sum( grad_real**2, dim=-1 ) )
        
        return 1 - torch.mean( prod/(fake_norm*real_norm) )

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


# Instantiate a model and dataset
net = ResNetUNet_V2(1)


depths = np.load('Data_management/dataset.npy').item()
#depths = ['Test_samples/frame-000000.depth.pgm','Test_samples/frame-000025.depth.pgm','Test_samples/frame-000050.depth.pgm','Test_samples/frame-000075.depth.pgm']
dataset = Dataset(depths['train'])
dataset_val = Dataset(depths['val'],train = False)

# dataset = Dataset(np.load('Data_management/dataset.npy').item()['train'][1:20])
# Parameters
params = {'batch_size': 16 ,
          'shuffle': True,
          'num_workers': 12,
          'pin_memory': True}
params_test = {'batch_size': 16 ,
          'shuffle': False,
          'num_workers': 12,
          'pin_memory': True}
training_generator = data.DataLoader(dataset,**params)
val_generator = data.DataLoader(dataset_val,**params_test)

net.train()
print(net)


# Loss
depth_criterion = RMSE_log()
grad_loss = GradLoss()
normal_loss = NormalLoss()
mani_loss = RMSE()
# Use gpu if possible and load model there
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = net.to(device)

# Optimizer
optimizer_ft = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
#scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
loss_list = []
history_val = []
best_loss = 50
for epoch in range(25):
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
        outputs = depths.cuda()
        # Clean grads
        optimizer_ft.zero_grad()

        #Forward
        predicts, grads, manifolds = net(inputs,outputs)
           
        #Backward+update weights
        depth_loss = depth_criterion(predicts[0], outputs) #+ depth_criterion(predicts[1], outputs)
        
        # Grad loss
        gradie_loss = 0.
        if epoch > 4:
            real_grad = net.imgrad(outputs)
            gradie_loss = grad_loss(grads[0], real_grad)#+ grad_loss(grads[1], real_grad)
        #normal_loss = normal_loss(predict_grad, real_grad) * (epoch>7)

        # Manifold loss
        embed_lose = 0#mani_loss(manifolds[0],manifolds[1])

        loss = depth_loss + 50*gradie_loss #+0.045*embed_lose# + normal_loss
        loss.backward()
        optimizer_ft.step()
        loss_train+=loss.item()*inputs.size(0)
        if cont%250 == 0:
            #loss.append(depth_loss.item())
            print("TRAIN: [epoch %2d][iter %4d] loss: %.4f" \
            % (epoch, cont, depth_loss.item()))
            print("Mani: {}, depth:{}, gradient{}".format(0.075*embed_lose, depth_loss, gradie_loss))
    if epoch%1==0:
        predict_depth = predicts[0].detach().cpu()
        #np.save('pspnet'+str(epoch), saver)
        save_predictions(predict_depth[0].detach(), rgbs[0], outputs[0],name ='unet_train1_epoch_'+str(epoch))
        #predict_depth = predicts[1].detach().cpu()
        #np.save('pspnet'+str(epoch), saver)
        save_predictions(predict_depth[1].detach(), rgbs[1], outputs[1],name ='unet_train2_epoch_'+str(epoch))


    loss_train = loss_train/dataset.__len__()
    print("\n FINISHED TRAIN epoch %2d with loss: %.4f " % (epoch, loss_train ))
    # Val
    loss_list.append(loss_train)
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
            predicts, grads, manifolds= net(inputs,outputs)

            #Sobel grad estimates:
            real_grad = net.imgrad(outputs)

            depth_loss = depth_criterion(predicts[0], outputs)#+depth_criterion(predict_grad, real_grad)
            loss_val+=depth_loss.item()*inputs.size(0)
            if cont%250 == 0:
                print("VAL: [epoch %2d][iter %4d] loss: %.4f" \
                % (epoch, cont, depth_loss))   
            
            #scheduler.step()
        if epoch%1==0:
            predict_depth = predicts[0].detach().cpu()
            #np.save('pspnet'+str(epoch), saver)
            save_predictions(predict_depth[0].detach(), rgbs[0], outputs[0],name ='unet1_epoch_'+str(epoch))
            #predict_depth = predicts[1].detach().cpu()
            #np.save('pspnet'+str(epoch), saver)
            save_predictions(predict_depth[1].detach(), rgbs[1], outputs[1],name ='unet2_epoch_'+str(epoch))


        loss_val = loss_val/dataset_val.__len__()
        history_val.append(loss_val)
        print("\n FINISHED VAL epoch %2d with loss: %.4f " % (epoch, loss_val ))
        
    if loss_val< best_loss and epoch>6:
        best_loss = depth_loss
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(net.state_dict(), 'model_unet_V2')





np.save('loss_unet',loss_list)
np.save('loss_val_unet',history_val)
