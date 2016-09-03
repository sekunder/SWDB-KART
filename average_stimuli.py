import data_preprocessing as dp
import fourier_transform as ft
import image_handling as ih
import numpy as np
from scipy.ndimage import imread


# the plan for this script is to grab the average natural scene stimulus that excites a neuron,
# compute its fourier transform, and then compare that to the fourier transformed static gratings.

def get_pref_ori_image(cell_specimen_id):
	pass

# image = ih.load_image_from_file('/Users/sekunder/python/SWDB/images/NS_stimulus_images/NS_0.png')

def compare_fourier(im1, im2):
	# TODO what actual comparison do I want to do?
	# Answer: add a function to the fourier_transform package that computes the "orientation" of an image.
	# Then figure out what we're doing with it.
	o1 = ft.image_orientation(im1)
	o2 = ft.image_orientation(im2)
	return abs(o1 - o2)

# Scratchwork/debugging

sweet_image_descriptions = {33:"puppies!",
				55:"caw caw, motherfucker",
				57:"ugly duck",
				81:"wood paneling",
				113:"pencils vertical",
				114:"pencils horizontal",
				115:"georgia o'keefe"}

sweet_images = [
	{"id" : img_id,
	 "description" : description,
	 "image" : imread('/Users/sekunder/python/SWDB/SWDB-KART/images/NS_stimulus_images/NS_' + str(img_id) + '.png',
					  flatten = True)
	} for img_id,description in sweet_image_descriptions.items()
	]

pct_trials = [0.01, 0.03, 0.05, 0.07, 0.09]
for d in sweet_images:
	for pct in pct_trials:
		d["orientation " + str(pct)] = ft.image_orientation(d["image"])

v_pencils_dict = sweet_images[4]
for pct in pct_trials:
	print('Orientation, computed using ' + str(int(100*pct)) + '% threshold: ' + str(v_pencils_dict["orientation " + str(pct)]))