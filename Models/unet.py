# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, depth=5, wf=6, padding=True,
                 batch_norm=True, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        filter_per_layer = [ 64, 64,  64, 128, 128,256, 256]
        self.input_cnn = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, bias = False,padding =1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace = True),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                                   )
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()

        for i in range(depth):
            self.down_path.append(UNetConvBlock(filter_per_layer[i], filter_per_layer[i+1],
                                                padding, batch_norm))
            prev_channels = filter_per_layer[i+1]
        filter_per_layer = [ 64, 64,  64, 128, 192,256, 256]

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, filter_per_layer[i], up_mode,
                                            padding, batch_norm))
            prev_channels = filter_per_layer[i]

        self.last = nn.Sequential(nn.Conv2d(prev_channels, n_classes, kernel_size=1),
                    nn.Sigmoid())
        self.x_sobel, self.y_sobel = self.make_sobel_filters()
        self.x_sobel = self.x_sobel.cuda()
        self.y_sobel = self.y_sobel.cuda()


    def forward(self, x):
        blocks = []
        x = self.input_cnn(x)
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        p = self.last(x)
        return p, self.imgrad(p)
    
    def make_sobel_filters(self):
        ''' Returns sobel filters as part of the network'''

        a = torch.Tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

        # Add dims to fit batch_size, n_filters, filter shape
        a = a.view((1,1,3,3))
        a = Variable(a)

        # Repeat for vertical contours
        b = torch.Tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

        b = b.view((1,1,3,3))
        b = Variable(b)

        return a,b

    
    def imgrad(self,img):
        # Filter horizontal contours
        G_x = F.conv2d(img, self.x_sobel)
        
        # Filter vertical contrours
        G_y = F.conv2d(img, self.y_sobel)

        G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
        return G




class UNet_V2(nn.Module):
    def __init__(self, in_channels=(3,1), n_classes=1, depth=5, wf=6, padding=True,
                 batch_norm=True, up_mode='upconv', map_depth_channels = True):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (tupple): number of input channels per encoder
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.map_depth_channels = map_depth_channels
        self.depth = depth
        prev_channels = in_channels[0]

        self.input_cnn = nn.Conv2d(in_channels[1], in_channels[0], kernel_size=1)

        # RGB Encoder
        self.down_path_RGB = nn.ModuleList()
        for i in range(depth):
            self.down_path_RGB.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)
        

        # Depth encoder
        prev_channels_depth = in_channels[1]

        # Trick to load pretrained_resnet
        if map_depth_channels:
            prev_channels_depth = 3


        self.down_path_depth = nn.ModuleList()
        for i in range(depth):
            self.down_path_depth.append(UNetConvBlock(prev_channels_depth, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels_depth = 2**(wf+i)

        # Check dimensions
        assert prev_channels_depth == prev_channels

        # Common?? decoder
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        # Output
        self.last = nn.Sequential(nn.Conv2d(prev_channels, n_classes, kernel_size=1),
                    nn.Sigmoid())

        # Sobel filters
        self.x_sobel, self.y_sobel = self.make_sobel_filters()
        self.x_sobel = self.x_sobel.cuda()
        self.y_sobel = self.y_sobel.cuda()


    def forward(self, x, x_depth):
        if map_depth_channels:
            x_depth = input_cnn(x_depth)
        # Encoder RGB
        blocks_RGB = []
        for i, down in enumerate(self.down_path_RGB):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks_RGB.append(x)
                x = F.max_pool2d(x, 2)

        # Encoder depth
        for i, down in enumerate(self.down_path_depth):
            x_depth = down(x_depth)
            if i != len(self.down_path)-1:
                x_depth = F.max_pool2d(x_depth, 2)

        # Decoder RGB
        for i, up in enumerate(self.up_path):
            x = up(x, blocks_RGB[-i-1])


        # Decoder Depth
        for i, up in enumerate(self.up_path):
            x_depth = up(x_depth, blocks_RGB[-i-1])

        p_RGB = self.last(x)
        p_depth = self.last(x_depth)
        return p_RGB, self.imgrad(p_RGB), p_depth , self.imgrad(p_depth)
    
    def make_sobel_filters(self):
        ''' Returns sobel filters as part of the network'''

        a = torch.Tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

        # Add dims to fit batch_size, n_filters, filter shape
        a = a.view((1,1,3,3))
        a = Variable(a)

                # Repeat for vertical contours
        b = torch.Tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

        b = b.view((1,1,3,3))
        b = Variable(b)

        return a,b

    
    def imgrad(self,img):
        # Filter horizontal contours
        G_x = F.conv2d(img, self.x_sobel)
        
        # Filter vertical contrours
        G_y = F.conv2d(img, self.y_sobel)

        G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
        return G


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []
        kernel_size = 3
        block.append(nn.Conv2d(in_size, out_size, kernel_size=kernel_size,
                               padding=int(padding), bias=False))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU(inplace=True))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding), bias=False))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        print(up.size(), crop1.size())
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out



if __name__ == "__main__":        
    unet = UNet()
    print(unet)