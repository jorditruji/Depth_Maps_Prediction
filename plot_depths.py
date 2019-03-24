import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from Data_management.dataset import Dataset
from Data_management.depth_preprocessing import read_depth, process_depth

img0 = np.squeeze(np.load('Test_outputs/unet2.npy'))


img50 = np.squeeze(np.load('v3_pred0.npy'))




print(img50.shape)
depths = ['Test_samples/frame-000000.depth.pgm','Test_samples/frame-000025.depth.pgm','Test_samples/frame-000050.depth.pgm','Test_samples/frame-000075.depth.pgm']
dataset = Dataset(depths)

print("Testing depth2RGB")
rgb_frames = dataset.depth2RGB()
print("Translated '{}' for '{}'".format(depths[0].split('/')[-1],rgb_frames[0].split('/')[-1]))

# Test depth reader
depth = read_depth(depths[-1])
processed_depth, mask, real_depth = process_depth(depth,1)
# Test jpg reader
rgb_2 = dataset.read_jpg(rgb_frames[-1])

#Plot results
f, axarr = plt.subplots(2, 2)
axarr[0,0].imshow(rgb_2)#, 'gray', interpolation='nearest')

axarr[0,0].set_title('Original RGB')

axarr[0,1].imshow(processed_depth,'gray',interpolation='nearest')
axarr[0,1].set_title('Processed Depth')

axarr[1,0].imshow(img0,'gray', interpolation='nearest')
axarr[1,0].set_title('50 epochs')

axarr[1,1].imshow(np.transpose(np.swapaxes(img50,0,-1)))#,'gray', interpolation='nearest')
axarr[1,1].set_title('100 epochs')
plt.show()
print("Finished tests")
