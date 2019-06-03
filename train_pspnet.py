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
dataset_val = Dataset(depths['val'],train = False)

# dataset = Dataset(np.load('Data_management/dataset.npy').item()['train'][1:20])
# Parameters
params = {'batch_size': 36 ,
          'shuffle': True,
          'num_workers': 16,
          'pin_memory': True}
params_Val = {'batch_size': 36 ,
          'shuffle': False,
          'num_workers': 16,
          'pin_memory': True}
training_generator = data.DataLoader(dataset,**params)
val_generator = data.DataLoader(dataset_val,**params_Val)

net.train()
print(net)


# Loss
depth_criterion = RMSE_log()
grad_loss = GradLoss()
normal_loss = NormalLoss()
# Use gpu if possible and load model there
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = net.to(device)

# Optimizer
optimizer_ft = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
#scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
loss_list = []
mse_list = []
grad_list = []
history_val = []
best_loss = 50
for epoch in range(25):
    # Train
    net.train()
    cont = 0
    loss_train = 0.0
    log_mse = 0.0
    grad_loss = 0.0

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
        

        #Backward+update weights
        depth_loss = depth_criterion(predict_depth, outputs)
        gradie_loss = 0.
        if epoch > 4:
            real_grad = net.imgrad(outputs)
            gradie_loss = grad_loss(predict_grad, real_grad)
            grad_loss += gradie_loss*inputs.size(0)
        #normal_loss = normal_loss(predict_grad, real_grad) * (epoch>7)
        loss = depth_loss + 12*gradie_loss# + normal_loss
        loss.backward()
        optimizer_ft.step()
        loss_train+=loss.item()*inputs.size(0)
        log_mse+=depth_loss.item()*inputs.size(0)


        if cont%250 == 0:
            #loss.append(depth_loss.item())
            print("TRAIN: [epoch %2d][iter %4d] log_MSEloss: %.4f" \
            % (epoch, cont, depth_loss.item()))
            #loss.append(depth_loss.item())
            if epoch>4:
                print("TRAIN: [epoch %2d][iter %4d] log_GRADloss total: %.4f" \
                % (epoch, cont, gradie_loss.item()))
    if epoch%2==0:
        predict_depth = predict_depth.detach().cpu()
        #np.save('pspnet'+str(epoch), saver)
        save_predictions(predict_depth[0].detach(), rgbs[0], outputs[0],name ='pspnet_train_epoch_'+str(epoch))


    loss_train = loss_train/dataset.__len__()
    grad_loss = grad_loss/dataset.__len__()
    log_mse = log_mse/dataset.__len__()

    print("\n FINISHED TRAIN epoch %2d with loss: %.4f " % (epoch, loss_train ))
    # Val
    loss_list.append(loss_train)
    mse_list.append(log_mse)
    grad_list.append(grad_loss)
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

            depth_loss = depth_criterion(predict_depth, outputs)#+depth_criterion(predict_grad, real_grad)
            loss_val+=depth_loss.item()*inputs.size(0)
            if cont%250 == 0:
                print("VAL: [epoch %2d][iter %4d] loss: %.4f" \
                % (epoch, cont, depth_loss))   
            
            #scheduler.step()
        if epoch%2==0:
            predict_depth = predict_depth.detach().cpu()
            #np.save('pspnet'+str(epoch), saver)
            save_predictions(predict_depth[0].detach(), rgbs[0], outputs[0],name ='pspnet_epoch_'+str(epoch))


        loss_val = loss_val/dataset_val.__len__()
        history_val.append(loss_val)
        print("\n FINISHED VAL epoch %2d with loss: %.4f " % (epoch, loss_val ))
        
    if loss_val< best_loss and epoch>6:
        best_loss = depth_loss
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(net.state_dict(), 'model_pspnet_V2')



np.save('loss_psp',loss_list)
np.save('loss_val_psp',history_val)


np.save('loss_psp_mse',log_mse)
np.save('loss_psp_grad',grad_loss)
