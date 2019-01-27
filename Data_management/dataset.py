from __future__ import division
import torch
from torch.utils import data
import numpy as np
import time
from skimage import io, transform
from depth_preprocessing import read_depth


class Dataset(data.Dataset):
    """
    Class Dataset:
    - Parameters:
        list_IDs: Vector of depth image paths
    """
    def __init__(self, depth_names):
        # Paths to dataset samples
        self.depth_frames = depth_names
        self.RGB_frames = self.depth2RGB()

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.depth_frames)


    def __getitem__(self, index):
        '''Generates one sample of data'''
        start_time = time.time()
        # Select sample
        depth = read_depth(self.depth_frames[index])
        rgb = self.read_jpg(self.RGB_frames[index])
        return depth, RGB

    def depth2RGB(self):
        '''Edit strings to match rgb paths'''
        return [depth.replace('depth.pgm','color.jpg ') for depth in self.depth_frames]


    def read_jpg(file):
        '''Read and preprocess pgm depth map'''
        return io.imread(file)

if __name__ == '__main__':
    # Testing:
    # Sample data
    depths = ['/projects/world3d/2017-06-scannet/scene0289_01/frame-000525.depth.pgm', '/projects/world3d/2017-06-scannet/scene0289_01/frame-000825.depth.pgm', '/projects/world3d/2017-06-scannet/scene0289_01/frame-000750.depth.pgm', '/projects/world3d/2017-06-scannet/scene0289_01/frame-000800.depth.pgm', '/projects/world3d/2017-06-scannet/scene0289_01/frame-001000.depth.pgm', '/projects/world3d/2017-06-scannet/scene0289_01/frame-000850.depth.pgm', '/projects/world3d/2017-06-scannet/scene0289_01/frame-001075.depth.pgm', '/projects/world3d/2017-06-scannet/scene0289_01/frame-000650.depth.pgm', '/projects/world3d/2017-06-scannet/scene0289_01/frame-000025.depth.pgm', '/projects/world3d/2017-06-scannet/scene0289_01/frame-000575.depth.pgm']
    dataset = Dataset(depths)

    print("Testing depth2RGB")
    rgb_frames = dataset.depth2RGB()
    print("Translated '{}' for '{}'").format(depths[0].split('/')[-1],rgb_frames[0].split('/')[-1])

    # Test depth reader

    # Test jpg reader