import skimage.io as io
import numpy as np
import h5py

def get_name(index, hdf5_data):
	title = hdf5_data[hdf5_data['scenes'][0][index]]
	return ''.join([chr(v[0]) for v in title[()]])

# data path
path_to_depth = '/home/jordi/Desktop/Datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat'

file = open("train_scenes.txt", "r") 
scenes = file.readlines()
scenes = [scene[:-1] for scene in scenes]
# read mat file

f = h5py.File(path_to_depth)

n_samples = f['images'].shape[0]
train_idxs = []
for i in range(n_samples):
	name = get_name(i, f)
	print(name, name in scenes)

	if name in scenes:
		train_idxs.append(i)

print(len(train_idxs))
'''
for i in range(1000):
	get_name(i, f)



# read 0-th image. original format is [3 x 640 x 480], uint8
img = f['images'][0]

# reshape
img_ = np.empty([480, 640, 3])
img_[:,:,0] = img[0,:,:].T
img_[:,:,1] = img[1,:,:].T
img_[:,:,2] = img[2,:,:].T

# imshow
img__ = img_.astype('float32')
io.imshow(img__/255.0)
io.show()


# read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
depth = f['depths'][0]

# reshape for imshow
depth_ = np.empty([480, 640, 3])
depth_[:,:,0] = depth[:,:].T
depth_[:,:,1] = depth[:,:].T
depth_[:,:,2] = depth[:,:].T

io.imshow(depth_/4.0)
io.show()
'''