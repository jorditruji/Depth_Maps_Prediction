from __future__ import division
import torch
from torch.utils import data
import numpy as np
import time
from skimage import io, transform
from depth_preprocessing import read_depth, process_depth


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
        return [depth.replace('depth.pgm','color.jpg') for depth in self.depth_frames]


    def read_jpg(self,file):
        '''Read and preprocess pgm depth map'''
        return io.imread(file)

if __name__ == '__main__':
    # Testing:
    # Sample data
    from matplotlib import pyplot
    depths = ['../Test_samples/frame-000000.depth.pgm','../Test_samples/frame-000025.depth.pgm','../Test_samples/frame-000050.depth.pgm','../Test_samples/frame-000075.depth.pgm']
    dataset = Dataset(depths)

    print("Testing depth2RGB")
    rgb_frames = dataset.depth2RGB()
    print("Translated '{}' for '{}'").format(depths[0].split('/')[-1],rgb_frames[0].split('/')[-1])

    # Test depth reader
    depth = read_depth(depths[-1])
    processed_depth, mask, real_depth = process_depth(depth,1)
    # Test jpg reader
    rgb = dataset.read_jpg(rgb_frames[-1])

    #Plot results
    f, axarr = pyplot.subplots(2, 2)
    axarr[0,0].imshow(depth, 'gray', interpolation='nearest')
    axarr[0,0].set_title('Original depth')
    axarr[0,1].imshow(rgb)
    axarr[0,1].set_title('Original RGB')
    axarr[1,0].imshow(processed_depth,'gray', interpolation='nearest')
    axarr[1,0].set_title('Navier Stokes Impaint')
    axarr[1,1].imshow(np.squeeze(mask),'gray', interpolation='nearest')
    axarr[1,1].set_title('MASK')
    pyplot.show()
    print "Finished tests"
