from scipy import ndimage
import cv2
import numpy as np
from sklearn.cluster import KMeans


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def apply_gaussian_filter(image_arr, sigma):
        smoothed_image = ndimage.gaussian_filter(image_arr, sigma=sigma)
        return smoothed_image

    @staticmethod
    def apply_median_filter(image_arr, sigma):
        smoothed_image = ndimage.median_filter(image_arr, sigma)
        return smoothed_image

    @staticmethod
    def apply_convolve_filter(image_arr, kernel_size, sigma):
        # Gaussian kernel
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)
            ),
            (kernel_size, kernel_size)
        )
        kernel /= np.sum(kernel)
        result_image = np.zeros_like(image_arr, dtype=float)

        for i in range(image_arr.shape[2]):
            result_image[:, :, i] = ndimage.convolve(input=image_arr[:, :, i], weights=kernel)

        return result_image

    @staticmethod
    def threshold_dog_convolve(edge_img, img, kernel_size, sigma1, sigma2, threshold):
        # Generate Gaussian kernels
        g1 = ImageProcessor.gaussian_kernel(kernel_size, sigma=sigma1)
        g2 = ImageProcessor.gaussian_kernel(kernel_size, sigma=sigma2)
        dog = g1 - g2

        # Apply convolution smoothing to each channel
        for i in range(img.shape[2]):
            edge_img[:, :, i] = ndimage.convolve(input=img[:, :, i], weights=dog)

        # Create a binary image based on the threshold
        threshold_img = np.where(edge_img > threshold, 1, 0)

        return threshold_img

    @staticmethod
    def gaussian_kernel(kernel_size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2) / (2 * sigma ** 2)),
            (kernel_size, kernel_size)
        )
        normal = kernel / np.sum(kernel)
        return normal

    @staticmethod
    def threshold_dog_gaussian(img, sigma1, sigma2, threshold):
        # Calculate the DoG filter
        dog = (ImageProcessor.apply_gaussian_filter(img, sigma=sigma1) -
               ImageProcessor.apply_gaussian_filter(img, sigma=sigma2))

        # Create a binary image based on the threshold
        threshold_img = np.where(dog > threshold, 1, 0)

        return threshold_img

    @staticmethod
    def quantization_hsv(smooth_img, hsv_num):
        hsv = cv2.cvtColor(smooth_img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * float(hsv_num)  # (V)alue Change
        quantized_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return quantized_image

    @staticmethod
    def quantization_lab(smooth_img, lab_num):
        lab = cv2.cvtColor(smooth_img, cv2.COLOR_RGB2Lab)
        lab[:, :, 0] = lab[:, :, 0] * float(lab_num)  # (L)uminance Change
        quantized_image = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        return quantized_image

    @staticmethod
    def quantization_KMeans(image, num_colors=16):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        height, width, _ = lab_image.shape
        pixels = lab_image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)
        quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
        quantized_image = quantized_pixels.reshape(height, width, 3)
        quantized_image = cv2.cvtColor(quantized_image.astype(np.uint8), cv2.COLOR_Lab2BGR)

        return quantized_image
