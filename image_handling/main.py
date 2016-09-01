import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def plot_mean_image(cell_specimen_id,ns,images,images_arr,traces_arr,thresh=0.3,weighted=False):
	# cell_idx = np.where(ns.cell_id==cell_specimen_id)[0][0]
	# thresh_inds = np.where(traces_arr[cell_idx,:]>=thresh)[0]
	# thresh_inds -= 6
	# thresh_images = images_arr[thresh_inds]
	# n_images = len(np.unique(thresh_images))
	# img_stack = np.empty((thresh_images.shape[0],images[0,:,:].shape[0],images[0,:,:].shape[1]))
	#
	# thresh_vals = traces_arr[cell_idx][thresh_inds]
	#
	# for i,img in enumerate(thresh_images):
	# 	img_stack[i,:,:] = images[img,:,:]
	img_stack,thresh_vals,n_images = get_calcium_triggered_image_stack(cell_specimen_id, ns, images, images_arr, traces_arr, thresh)

	if weighted:
		mean_img = mean_image(img_stack,thresh_vals)
		# mean_img = np.average(img_stack,axis=0,weights=thresh_vals)
	else:
		mean_img = mean_image(img_stack)
		# mean_img = np.mean(img_stack,axis=0)

	fig,ax=plt.subplots()
	ax.imshow(mean_img,cmap='gray')
	if weighted:
		ax.set_title('cell '+str(cell_specimen_id)+', weighted mean of '+str(n_images)+' images')
	else:
		ax.set_title('cell '+str(cell_specimen_id)+', mean of '+str(n_images)+' images')


def mean_image(img_stack,weights=None):
	"""Compute the mean image of an image stack, which is a 3 dimensional ndarray; the first index is the image index and the other two are pixels.
If weights are supplied, takes the weighted average. Returns a single image which is the (weighted) mean."""
	if weights is None:
		ans = np.mean(img_stack,axis=0)
	else:
		ans = np.average(img_stack,axis=0,weights=weights)
	return ans

def get_calcium_triggered_image_stack(cell_specimen_id, ns, images, images_arr,traces_arr,thresh=0.3):
	"""Returns a stack of images that seem to trigger responses"""
	cell_idx = np.where(ns.cell_id == cell_specimen_id)[0][0]
	thresh_inds = np.where(traces_arr[cell_idx,:] >= thresh)[0]
	thresh_inds -= 6
	thresh_vals = traces_arr[cell_idx][thresh_inds]
	thresh_images = images_arr[thresh_inds]
	n_images = len(np.unique(thresh_images))
	img_stack = np.empty((thresh_images.shape[0], images[0, :, :].shape[0], images[0, :, :].shape[1]))
	for i, img in enumerate(thresh_images):
		img_stack[i, :, :] = images[img, :, :]
	return img_stack,thresh_vals,n_images

def gaussian_blur(img,sigma=19.6):
	"""Applies a gaussian blur to given image and returns the result. By default, sigma is 19.6, which is roughly equal, in pixels, to 2 degrees of the mouse's visual field (ignoring the warping caused by using a flat screen)"""
	return gaussian_filter(img,sigma)