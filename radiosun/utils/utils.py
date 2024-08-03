import numpy as np
import pywt


def SunSolidAngle(R):
    R_deg = (R / 3600)
    return np.pi * (np.pi * R_deg / 180) ** 2

def gauss2d(x, y, amplitude_x, amplitude_y, mean_x, mean_y, sigma_x, sigma_y):
    x, y = np.meshgrid(x, y)
    g = amplitude_x * amplitude_y * np.exp(-((x - mean_x) ** 2 / (2 * sigma_x ** 2) + (y - mean_y) ** 2 / (2 * sigma_y ** 2)))
    return g

def gauss1d(x, amplitude, mean, sigma):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def gaussian_mixture(params, x, y):
    model = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude = params[i]
        mean = params[i + 1]
        stddev = params[i + 2]
        model += gauss1d(x, amplitude, mean, stddev)
    return y - model

def create_rectangle(size, width, height):
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    x, y = np.meshgrid(x, y)
    rectangle = np.zeros((size, size))
    mask = (abs(x) <= width/2) & (abs(y) <= height/2)
    rectangle[mask] = 1
    return rectangle

def create_sun_model(size, radius):
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    mask = x ** 2 + y ** 2 <= radius ** 2
    sun_model = np.zeros((size, size))
    sun_model[mask] = 1
    return sun_model

def calculate_area(image):
    return np.trapz(image)

def bwhm_to_sigma(bwhm):
    fwhm = np.sqrt(1 / 2) * bwhm
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def flip_and_concat(values, flip_values=False):
    flipped = -values[::-1][:-1] if flip_values else values[::-1][:-1]
    return np.concatenate((flipped, values))
                          
def error(scale_factor, experimental_data, theoretical_data):
   return np.sum((scale_factor * experimental_data - theoretical_data) ** 2)

def wavelet_denoise(data, wavelet, level):
   N = 2**np.ceil(np.log2(len(data))).astype(int)
   data_padded = np.pad(data, (0, N - len(data)), 'constant', constant_values=(0, 0))
   coeff = pywt.wavedec(data_padded, wavelet, mode="sym")
   sigma = np.median(np.abs(coeff[-level] - np.median(coeff[-level]))) / 0.6745
   uthresh = sigma * np.sqrt(2 * np.log(len(data_padded)))
   coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
   return pywt.waverec(coeff, wavelet, mode="sym")[:len(data)]
