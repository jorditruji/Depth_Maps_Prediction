from __future__ import division
import torch
from torch.utils import data
import numpy as np
import time
from .depth_preprocessing import read_depth, process_depth
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


'RECORDATORI SCANNET dataset: \
Max depth(uint16): 9998 \
Min depth(uint16) : 264 \
Mean RGB: [0.4944742  0.4425867  0.38153833] \
Std RGB: [0.23055981 0.22284868 0.21425385] '

class Dataset(data.Dataset):
    """
    Class Dataset:
    - Parameters:
        depth_names: Vector of depth image paths
    """
    def __init__(self, depth_names, train = True):
        # Paths to dataset samples
        self.train = train
        self.depth_frames = depth_names
        self.RGB_frames = self.depth2RGB()
        self.RGB_transforms_train = transforms.Compose([transforms.Resize((240,320))
                                                ,transforms.ColorJitter()
                                                ,transforms.ToTensor()                                                
                                                ,transforms.Normalize([0.4944742, 0.4425867, 0.38153833], [0.23055981, 0.22284868, 0.21425385])
                                                ])
        self.RGB_transforms_test = transforms.Compose([transforms.Resize((480,640))
                                                ,transforms.ToTensor()                                                
                                                ,transforms.Normalize([0.4944742, 0.4425867, 0.38153833], [0.23055981, 0.22284868, 0.21425385])
                                                ])
        self.depth_transforms = transforms.Compose([transforms.Resize((240,320)),
                                                    transforms.ToTensor()
                                                 #Need to means and stds
                                                 #,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.depth_frames)

    '''
    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        depth = read_depth(self.depth_frames[index])
        depth = process_depth(depth)
        # Format #channels, H, W

        rgb = self.read_jpg_train(self.RGB_frames[index])
        depth= self.depth_transforms(Image.fromarray(depth, mode = 'L'))


        return depth, rgb
    '''
    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        depth = read_depth(self.depth_frames[index])
        depth = process_depth(depth)
        # Format #channels, H, W

        rgb = self.read_jpg_train(self.RGB_frames[index])
        depth= self.depth_transforms(Image.fromarray(depth, mode = 'L'))


        return depth, rgb, self.depth_frames[index]

    def depth2RGB(self):
        '''Edit strings to match rgb image paths'''
        return [depth.replace('depth.pgm','color.jpg') for depth in self.depth_frames]


    def read_jpg_train(self,file):
        '''Read and preprocess rgb frame'''
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(file, 'rb') as f:
            img = Image.open(f).convert('RGB')
        # Apply transforms depending on model status
        if self.train:
            img = self.RGB_transforms_train(img)
        else:
            img = self.RGB_transforms_test(img)
        return img


    def read_jpg(self,file):
        '''Read and preprocess rgb frame'''
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(file, 'rb') as f:
            img = Image.open(f).convert('RGB')
        trans = transforms.Compose([transforms.Resize((480,640))])
        return img

    def imgrad(self,img):
        ''' Returns sobel gradient '''
        #Black and white input image x, 1x1xHxW

        #Uncomment for testing
        
        img = torch.Tensor(img/256)
        img.unsqueeze_(0)
        img.unsqueeze_(0)
        
        img = Variable(img)
        # Initializer sobel filters
        a = torch.Tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

        # Add dims to fit batch_size, n_filters, filter shape
        a = a.view((1,1,3,3))
        a = Variable(a)

        # Filter horizontal contours
        G_x = F.conv2d(img, a)

        # Repeat for vertical contours
        b = torch.Tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

        b = b.view((1,1,3,3))
        b = Variable(b)
        G_y = F.conv2d(img, b)

        G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
        return G


if __name__ == '__main__':
    # Testing:
    # Sample data
    from matplotlib import pyplot



    depths = ['../Test_samples/frame-000000.depth.pgm','../Test_samples/frame-000025.depth.pgm','../Test_samples/frame-000050.depth.pgm','../Test_samples/frame-000075.depth.pgm']
    dataset = Dataset(depths)

    print("Testing depth2RGB")
    rgb_frames = dataset.depth2RGB()
    print("Translated '{}' for '{}'".format(depths[0].split('/')[-1],rgb_frames[0].split('/')[-1]))

    # Test depth reader
    depth = read_depth(depths[-1])
    processed_depth = process_depth(depth,1)
    # Test jpg reader
    rgb = dataset.read_jpg_train(rgb_frames[-1])
    rgb_2 = dataset.read_jpg(rgb_frames[-1])
    print(rgb.size())


    # Test get_item
    sample_depth, sample_RGB = dataset.__getitem__(0)
    print("IMPORTANT!!!")
    print(sample_RGB.size(), sample_depth.size())

    # Matplotlib style display = channels last
    rgb= np.swapaxes(rgb.numpy(),0,-1)
    print(rgb.shape)
    rgb = np.swapaxes(rgb,0,1)
    print(rgb.shape)

    #Plot results
    f, axarr = pyplot.subplots(2, 2)
    axarr[0,0].imshow(rgb_2)#, 'gray', interpolation='nearest')
    axarr[0,0].set_title('Original RGB')
    axarr[0,1].imshow(rgb)
    axarr[0,1].set_title('Processed RGB')
    axarr[1,0].imshow(processed_depth,'gray', interpolation='nearest')
    axarr[1,0].set_title('Navier Stokes Impaint')


    # Test depth_gradient:
    gradient = dataset.imgrad(processed_depth)
    gradient = gradient.data.numpy()
    axarr[1,1].imshow(np.squeeze(gradient)/np.max(gradient),'gray', interpolation='nearest')
    axarr[1,1].set_title('Gradients Sobel')
    pyplot.show()
    print("Finished tests")
