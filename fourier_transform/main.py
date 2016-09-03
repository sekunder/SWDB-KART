import numpy as np
# import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import interp2d
from scipy.ndimage import map_coordinates
import radialProfile



def log_fourier_transform(image):
	"""Shifts the supplied image so its mean intensity is 0, then computes the log of the absolute value of the fourier transform.
	Input:
		:numpy.ndarray image : the image to transform
	Output:
		:numpy.ndarray shifted_image : the image with mean shifted to 0
		:numpy.ndarray fft_image : The image, passed through the following transformations (in order): shift mean to 0, FFT, FFTshift absolute value, log10.
			See scipy.fftpack.fft2 and scipy.fftpack.fftshift"""
	shifted_image = image - np.mean(image[:])
	shifted_fft_raw_image = fftpack.fftshift(fftpack.fft2(shifted_image))
	log_fft_image =  np.log10(np.abs(shifted_fft_raw_image))
	# F1 = fftpack.fft2(image - np.mean(image[:]))  # shift them quadrants around so dat low spatial frequencies are da center of da 2D fourier transformed image.
	# F2 = fftpack.fftshift(F1)  # Calculatin' a 2D power spectrum
	# Calculate the azimuthally averaged 1D power spectrum
	# psd2D = np.abs(shifted_fft_raw_image) ** 2
	# psd1D = radialProfile.azimuthalAverage(psd2D)
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
	return shifted_image, log_fft_image#, psd1D

def radial_image_intensity(image, x_0, y_0, r, theta=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4), num_interp_points=None):
	"""returns the image's 'radial intensity' around the given point, to a distance of r, for each angle in the iterable theta.
	Input:
		image, x, y, r, theta
	Output:
		theta : the input theta, or the default of 45 degree increments from 0 to 180 (inclusive)
		intensity : the intensity in the corresponding direction. The idea is you can just say plot(theta, intensity) and get something useful."""
	intensity = [image_line_segment_intensity(image, x_0, y_0, x_0 + r * np.cos(t), y_0 + r * np.sin(t), num_steps=num_interp_points) for t in theta]
	return theta, intensity

def image_line_segment_intensity(image, x_0, y_0, x_1, y_1, num_steps=None):
	"""computes the 'mean intensity' of the image along the line segment from (x0,y0) to (x1, y1).
	This is defined as 1/r times the sum of the pixel intensities on the interpolated line between the two endpoints (inclusive)

	Input:
		image : ndarray image
		x0, y0 : the (pixel) coordinates of the start point
		x1, y1 : the (pixel) coordinates of the end point

	Output:
		mean intensity

	Note:
		This is undoubtedly going to be rather crude, because we are interpolating from pixel to pixel. Time constraints being what they are, this is the best we're gonna get."""
	r = np.hypot(np.abs(x_0 - x_1), np.abs(y_0 - y_1))
	int_pixels,ds = interpolated_pixel_values(image, x_0, y_0, x_1, y_1, num_steps=num_steps)
	total_intensity =  np.sum(ds * int_pixels)
	# pretend we have an iterable collection called interpolated_pixels
	# for pixel in int_pixels:
	# 	total_intensity += image(pixel[0],pixel[1])
	return float(total_intensity) / r


def interpolated_pixel_values(image, x_0, y_0, x_1, y_1, num_steps=None, ord=1):
	"""returns the pixel values along the interpolated line between the (pixel) coordinates in the given image, using "floor" interpolation.
	Input:
		image : image as ndarray
		x_0, y_0 : start coordinates
		x_1, y_1 : end coordinates
		num_steps : number of sample points to take in the middle. By default, this is the ceiling of the distance between the start and end point
		ord : order of interpolation
	Output:
		pixel_values : the image intensity at each sample point
		ds : the step size, for taking Riemann sums
		"""
	if not num_steps:
		num_steps = 200.0
	dx = (x_1 - x_0)/num_steps
	dy = (y_1 - y_0)/num_steps
	ds = np.hypot(dx,dy)
	x_range = np.arange(x_0,x_1,dx)
	y_range = np.arange(y_0,y_1,dy)
	pixel_values = map_coordinates(image, np.vstack((x_range, y_range)),order=ord)
	# (Delta_x, Delta_y) = (y_1 - y_0, x_1 - x_0)
	# distance = np.hypot(np.abs(Delta_x), np.abs(Delta_y))
	# if not num_steps:
	# 	num_steps = np.ceil(distance)
	# # dx,dy = Delta_x/float(num_steps), Delta_y/float(num_steps)
	# ds = float(distance)/float(num_steps)
	# # x = np.arange(image.shape[1])
	# # y = np.arange(image.shape[0])
	# # if not interpolator:
	# # 	interpolator = interp2d(x, y, image)
	#
	# # extract values on line from x_0,y_0 to x_1,y_1
	# xvalues = np.floor(np.linspace(x_0, x_1, num_steps))
	# yvalues = np.floor(np.linspace(y_0, y_1, num_steps))
	# pixel_values = image[xvalues.astype(np.int),yvalues.astype(np.int)]
	# # pixels = interpolate_pixels_along_line(x_0, y_0, x_1, y_1)
	# # dx = np.diff(pixels)
	return pixel_values, ds

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