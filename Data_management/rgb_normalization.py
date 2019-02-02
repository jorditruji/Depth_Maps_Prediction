from dataset import Dataset
import numpy as np
from PIL import Image
from depth_preprocessing import read_depth, process_depth



train_dict = np.load('dataset.npy').item()['train']
depths = ['../Test_samples/frame-000000.depth.pgm','../Test_samples/frame-000025.depth.pgm','../Test_samples/frame-000050.depth.pgm','../Test_samples/frame-000075.depth.pgm']

means = []
stds = []
max_depth = []
min_depth = []
dataset = Dataset(train_dict)
for sample,depth in zip(dataset.RGB_frames,dataset.depth_frames):
    with open(sample, 'rb') as f:
        img = Image.open(f).convert('RGB')
    img = np.asarray(img)
    width_ ,height, n_channels = img.shape
    val = img.reshape(width_*height,n_channels)
    means.append(np.mean(val,axis=0))
    stds.append(np.std(val,axis=0))
    depth_img = read_depth(depth)
    max_depth.append( np.max(depth_img))
    min_depth.append(np.min(depth_img[depth_img>0]))

means = np.array(means)
stds = np.array(stds)

global_mean = np.mean(means,axis = 0)
global_std = np.mean(stds,axis = 0)

max_save = np.max(max_depth)
min_save = np.min(min_depth)


np.save ('means',global_mean)
np.save ('stds',global_std)
np.save ('max_save',max_save)
np.save ('means',min_save)