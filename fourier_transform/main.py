import numpy as np
# import matplotlib.pyplot as plt
from scipy import fftpack
# import radialProfile



def log_fourier_transform(image):
	"""Shifts the supplied image so its mean intensity is 0, then computes the log of the absolute value of the fourier transform.
	Input:
		:numpy.ndarray image : the image to transform
	Output:
		:numpy.ndarray shifted_image : the image with mean shifted to 0
		:numpy.ndarray fft_image : The image, passed through the following transformations (in order): shift mean to 0, FFT, FFTshift absolute value, log10.
			See scipy.fftpack.fft2 and scipy.fftpack.fftshift"""
	shifted_image = image - np.mean(image[:])
	transformed_image =  np.log10(np.abs(fftpack.fftshift(fftpack.fft2(shifted_image))))
	return shifted_image, transformed_image
	# F1 = fftpack.fft2(image - np.mean(image[:]))  # shift them quadrants around so dat low spatial frequencies are da center of da 2D fourier transformed image.
	# F2 = fftpack.fftshift(F1)  # Calculatin' a 2D power spectrum
	# psd2D = np.abs(F2) ** 2  # Calculate the azimuthally averaged 1D power spectrum
	# # psd1D = radialProfile.azimuthalAverage(psd2D, center=[459, 587])
	# log_image = np.log10(image + 1)
	# fft_image = np.log10(psd2D)
	# power_spectrum = radialProfile.azimuthalAverage(psd2D, center=[459, 587])
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
	# return log_image, fft_image, power_spectrum

def image_orientation(image,fraction_of_peak=0.05,degrees=True):
	"""computes an image's 'orientation' according to the following absurd metric:
	Computes the fourier transform of the image, then finds values greater than a given threshold, then fits a line to those points, and computes its slope.

	Input:
		image : 2-dimensional ndarray
		fraction_of_peak : Values below fraction_of_peak * np.max(np.abs(fftpack.fft2(shifted_image))) are ignored. Default is 0.05
		degrees : if true, convert output to degrees.
		"""
	# compute
	shifted_img = image - np.mean(image[:])
	fft_img = fftpack.fft2(shifted_img)
	fft_img = np.abs(fftpack.fftshift(fft_img))
	(M,N) = fft_img.shape
	#get indices greater than threshold
	theta = fraction_of_peak * np.max(fft_img)
	(x,y) = np.where(fft_img > theta)
	#translate indices to origin
	x -= M / 2
	y -= N / 2
	#find slope of best fit line
	m,_ = np.polyfit(x, y, 1)
	#get orientation = arctan(slope)
	ori = np.arccos(1 / (m ** 2 + 1))
	if degrees:
		ori *= 180/np.pi
	return ori