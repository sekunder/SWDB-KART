import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import radialProfile



def fourier_transform(image):
	"""Computes the fast fourier transform of the supplied image."""
	F1 = fftpack.fft2(image - np.mean(image[:]))  # shift them quadrants around so dat low spatial frequencies are da center of da 2D fourier transformed image.
	F2 = fftpack.fftshift(F1)  # Calculatin' a 2D power spectrum
	psd2D = np.abs(F2) ** 2  # Calculate the azimuthally averaged 1D power spectrum
	# psd1D = radialProfile.azimuthalAverage(psd2D, center=[459, 587])
	log_image = np.log10(image + 1)
	fft_image = np.log10(psd2D)
	power_spectrum = radialProfile.azimuthalAverage(psd2D, center=[459, 587])
	# if plot:
	# 	plt.figure(1)
	# 	plt.imshow(log_image, cmap="gray")  # The +1 shifts it to non-zero values
	# 	plt.figure(2)
	# 	plt.imshow(fft_image, cmap="jet")
	# 	plt.figure(3)
	# 	plt.semilogy(power_spectrum)
	# 	plt.title('1D Power Spectrum')
	# 	plt.xlabel('Spatial Frequency')
	# 	plt.ylabel('Power Spectrum')
	return log_image, fft_image, power_spectrum

def image_orientation(image):
	"""computes an image's 'orientation' according to [TODO: describe metric. Will involve computing FFT of image]"""
	# TODO what goes here?

	return 1.0