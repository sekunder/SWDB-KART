import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import radialProfile



def fourier_transform(image):
	"""Computes the fast fourier transform of the supplied image."""
	F1 = fftpack.fft2(image - np.mean(image[:]))  # shift them quadrants around so dat low spatial frequencies are da center of da 2D fourier transformed image.
	F2 = fftpack.fftshift(F1)  # Calculatin' a 2D power spectrum
	psd2D = np.abs(F2) ** 2  # Calculate the azimuthally averaged 1D power spectrum
	psd1D = radialProfile.azimuthalAverage(psd2D, center=[459, 587])
	# plottin dem results ya'll!
	img1 = np.log10(image + 1)
	img2 = np.log10(psd2D)
	img3 = psd1D
	if plot:
		plt.figure(1)
		plt.imshow(img1, cmap="gray")  # The +1 shifts it to non-zero values
		plt.figure(2)
		plt.imshow(img2, cmap="jet")
		plt.figure(3)
		plt.semilogy(img3)
		plt.title('1D Power Spectrum')
		plt.xlabel('Spatial Frequency')
		plt.ylabel('Power Spectrum')
	return img1, img2, img3

#
# image_transforms = {}  # this is a dictionary
# for i in range(natural_scenes.shape[0]):
# 	tmp = {}
# 	image = natural_scenes[i, :, :]
# 	img1, img2, img3 = fourier_transform(image)
# 	image_transforms[i] = {
# 		'fourier_trans_img': img1,
# 		'processed_image': img2,
# 		'power_spec': img3,  # when retrieving this use plt.semilogy as seen in the fouri_trans function
# 	}
# # the units are cycles per image


#show me dat natural scene!
#use this if you just want to look at a single image
# img = natural_scenes[113,:,:]
# plt.imshow(img,cmap= "gray")
# fouri_trans(img,plot=True)