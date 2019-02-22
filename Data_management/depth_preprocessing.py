# -*- coding: utf-8 -*-
from __future__ import division
import cv2
from read_pgm_depth import NetpbmFile
import numpy as np
from itertools import product
import time
from PIL import Image

'Contains the preprocessing functions of the depth maps like inpainting'



def equalize_hist(img):
	equ = cv2.equalizeHist(img)
	return equ

def read_depth(file):
	start_time = time.time()
	try:
		with NetpbmFile(file) as pam:
			img = pam.asarray(copy=False)
			#print("--- %s seconds reading depth frame---" % (time.time() - start_time))
			return img

	except ValueError as e:
		# raise  # enable for debugging
		print(file, e)


def process_depth_TFG(img):
	#BASURAAA
	zmax=np.max(img)
	norm_img=np.zeros(img.shape,dtype=np.uint8)
	mask=np.zeros(img.shape,dtype=np.uint8)
	mask_std=np.zeros(img.shape,dtype=np.uint8)
	cont=0
	h,w = img.shape
	for pos in product(range(h), range(w)):
		pixel =  img.item(pos[0],pos[1])
		if pixel>0:
			new_pix2=(float(pixel)/float(zmax))*254.0
			norm_img[pos]=255-new_pix2
			mask[pos]=0
		else:
			norm_img[pos]=0
			mask[pos]=255
		cont+=1

	dst_TELEA = cv2.inpaint(norm_img,mask,3,cv2.INPAINT_TELEA)
	#dst_TELEA=equalize_hist(dst_TELEA)

	return (dst_TELEA-255)*-1,mask
		

def process_depth(img,inpaint=0):
	# Converts depth information to uint8 gray scale and inpaints it
	start_time = time.time()
	#img=img/255.
	mask = img.copy()
	z_max = np.max(img)

	img = ((img/z_max)*255).astype('uint8')

	mask[mask==0]= 1
	mask[mask>1] = 0
	mask[mask>1] = 255
	mask=np.expand_dims(mask,-1)
	img = np.expand_dims(img,-1)
	mask = mask.astype('uint8')
	if inpaint == 0:
		processed_depth = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)
	else:
		processed_depth = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

	#dst_TELEA_inpainted=equalize_hist(dst_TELEA_inpainted)
	#real_depth = (processed_depth/255)*z_max
	#print("--- %s seconds processing depth frame---" % (time.time() - start_time))
	return processed_depth#, mask, real_depth





if __name__ == '__main__':

	# Tests
	from matplotlib import pyplot

	depth_sample = '../Test_samples/frame-000050.depth.pgm'
	RGB_sample = '../Test_samples/frame-000050.color.jpg'
	rgb_im = Image.open(RGB_sample)
	depth = read_depth(depth_sample)
	processed_depth = process_depth(depth,1)
	processed_depth_new = process_depth(depth,0)
	# Invertim prop de lluny
	#depth = 65535 - depth
	f, axarr = pyplot.subplots(2, 2)
	axarr[0,0].imshow(depth, 'gray', interpolation='nearest')
	axarr[0,0].set_title('Original depth')
	axarr[0,1].imshow(rgb_im)
	axarr[0,1].set_title('Original RGB')
	axarr[1,0].imshow(processed_depth_new,'gray', interpolation='nearest')
	axarr[1,0].set_title('Navier Stokes Impaint')
	axarr[1,1].imshow(processed_depth,'gray', interpolation='nearest')
	axarr[1,1].set_title('TELEA Impaint')
	pyplot.show()
	