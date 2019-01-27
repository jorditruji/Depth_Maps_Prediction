# -*- coding: utf-8 -*-
import cv2
from read_pgm_depth import NetpbmFile
import numpy as np
from matplotlib import pyplot


def read_depth(file):
	try:
		with NetpbmFile(file) as pam:
			img = pam.asarray(copy=False)
			return img

	except ValueError as e:
		# raise  # enable for debugging
		print(file, e)
		

def fill_holes(img):
	print "Holes"
	img=img/256
	mask = np.ones(img.shape)*img
	print type(mask)
	print np.unique(mask)

	mask[mask==0]= 1
	mask[mask>1] = 0
	print np.unique(mask)

	mask=np.expand_dims(mask,0)
	img = np.expand_dims(img,0)
	img.astype('uint8')
	mask.astype('uint8')
	print mask.shape
	print img.shape
	print type(mask)
	print np.unique(img)
	dst_TELEA = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)


if __name__ == '__main__':
	depth_sample = '../Test_samples/frame-000000.depth.pgm'
	RGB_sample = '../Test_samples/frame-000000.color.jpg'
	depth = read_depth(depth_sample)
	print depth[0][0]
	processed_depth = fill_holes(depth)
	# Invertim prop de lluny
	#depth = 65535 - depth
	pyplot.imshow(depth, 'gray', interpolation='nearest')
	pyplot.figure()
	pyplot.imshow(depth,'gray', interpolation='nearest')
	pyplot.show()
