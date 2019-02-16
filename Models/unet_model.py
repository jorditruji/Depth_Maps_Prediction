# full assembly of the sub-parts to form the complete net

from torch.autograd import Variable
import torch.nn.functional as F
import torch
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1):
        super(UNet, self).__init__()
        # Network itself
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        
        # Sobel filters
        self.x_sobel, self.y_sobel = self.make_sobel_filters()
        self.x_sobel = self.x_sobel.cuda()
        self.y_sobel = self.y_sobel.cuda()    

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

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)