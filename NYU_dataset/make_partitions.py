import skimage.io as io
import numpy as np
import h5py

'''
Max_depth:9.99547004699707
Min_depth: 0.7132995128631592
Mean_depth: [0.48261331 0.41353991 0.3988998 ]
Std: [0.2643962  0.27325641 0.2833328 ]

'''

def get_name(index, hdf5_data):
	title = hdf5_data[hdf5_data['scenes'][0][index]]
	return ''.join([chr(v[0]) for v in title[()]])



# Dataset_path + open of the db
path_to_depth = '/home/jordi/Desktop/Datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat'
f = h5py.File(path_to_depth)

# Read train scenes name
file = open("train_scenes.txt", "r") 
scenes = file.readlines()
scenes = [scene[:-1] for scene in scenes]

# amount of images in the dataset
n_samples = f['images'].shape[0]
train_idxs = []
means = []
stds = []
max_depth = []
min_depth = []
for i in range(n_samples):
	name = get_name(i, f)
	print(name, name in scenes)

	if name in scenes:
		train_idxs.append(i)
		rgb = f['images'][i] #[3 x 640 x 480], uint8
		depth = f['depths'][i] #[640 x 480], float64
		rgb = np.asarray(rgb)/255

		rgb = rgb.reshape(3, 640*480)
		print(rgb.shape, depth.shape)
		means.append(np.mean(rgb,axis=1))
		stds.append(np.std(rgb,axis=1))
		max_depth.append(np.max(depth))
		min_depth.append(np.amin(depth))



means = np.array(means)
stds = np.array(stds)

global_mean = np.mean(means,axis = 0)
global_std = np.mean(stds,axis = 0)

max_save = np.max(max_depth)


min_save = np.min(min_depth)

print ("Max:{}".format(max_save))
print ("Min: {}".format(min_save))
print ("Mean: {}".format(global_mean))
print ("Std: {}".format(global_std))

np.save ('means',global_mean)
np.save ('stds',global_std)
np.save ('max_save',max_save)
np.save ('min_save',min_save)
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