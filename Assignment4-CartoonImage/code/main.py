from ImageProcessor import ImageProcessor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def load_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is not None:
            return image
        else:
            return None
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None


def display_binary_image(image_array, title="Binary Image"):
    # 0-1 arrayini siyah-beyaz görüntü olarak çiz
    plt.imshow(image_array[:, :, 0], cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.show()


def save_image_3d_array(image_array, file_path):
    normalized_image = (image_array * 255).astype(np.uint8)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        # Save the image using OpenCV
        cv2.imwrite(file_path, image_array)
    except Exception as e:
        print(f"An error occurred: {e}")


def save_binary_image(image_array, file_path):
    binary_image = np.clip(image_array, 0, 1)
    scaled_image = (binary_image * 255).astype(np.uint8)
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        # Save the image using OpenCV
        cv2.imwrite(file_path, scaled_image)
    except Exception as e:
        print(f"An error occurred: {e}")


image_path = "data/Galata.png"  # Dosya yolu
image_arr = load_image(image_path)  # Img -> array

# -------------------------------------------------------------------------------------------------
# ------------------------------ Smoothed Part ---------------------------------------------------
# -------------------------------------------------------------------------------------------------

smoothed_image_gaussian_1 = ImageProcessor.apply_gaussian_filter(image_arr, 1)
# smoothed_image_gaussian_2 = ImageProcessor.apply_gaussian_filter(image_arr, 2)
# smoothed_image_gaussian_4 = ImageProcessor.apply_gaussian_filter(image_arr, 4)
# smoothed_image_gaussian_8 = ImageProcessor.apply_gaussian_filter(image_arr, 8)
#
save_image_3d_array(smoothed_image_gaussian_1, "result/galata/smoothed/gaussian_smoothed_1.png")
# save_image_3d_array(smoothed_image_gaussian_2, "result/galata/smoothed/gaussian_smoothed_2.png")
# save_image_3d_array(smoothed_image_gaussian_4, "result/galata/smoothed/gaussian_smoothed_4.png")
# save_image_3d_array(smoothed_image_gaussian_8, "result/galata/smoothed/gaussian_smoothed_8.png")
#
# smoothed_image_median = ImageProcessor.apply_median_filter(image_arr, 3)
# save_image_3d_array(smoothed_image_median, "result/galata/smoothed/median_smoothed.png")
#
# smoothed_image_convolve = ImageProcessor.apply_convolve_filter(image_arr, 4, 1)
# save_image_3d_array(smoothed_image_median, "result/galata/smoothed/convolve_smoothed.png")

# We continue with the smoothed_image_gaussian_1 image that we like.


# -------------------------------------------------------------------------------------------------
# ------------------------------ Edge Detection Part ---------------------------------------------
# -------------------------------------------------------------------------------------------------


# kernel_size, sigma1, sigma2, threshold = 5, 1, 2, 230
# threshold_img_convolve = ImageProcessor.threshold_dog_convolve(smoothed_image_gaussian_1, image_arr, kernel_size, sigma1, sigma2, threshold)
# filename = f"result/galata/edgeDetection/threshold_dog_convolve_{kernel_size}_{sigma1}_{sigma2}_{threshold}.png"
# save_binary_image(threshold_img_convolve, filename)


# kernel_size, sigma1, sigma2, threshold = 5,1,5,230
# threshold_img_convolve = ImageProcessor.threshold_dog_convolve(smoothed_image_gaussian_1, image_arr, kernel_size, sigma1, sigma2, threshold)
# filename = f"result/galata/edgeDetection/threshold_dog_convolve_{kernel_size}_{sigma1}_{sigma2}_{threshold}.png"
# save_binary_image(threshold_img_convolve, filename)


# kernel_size, sigma1, sigma2, threshold = 5,1,2,40
# threshold_img_convolve = ImageProcessor.threshold_dog_convolve(smoothed_image_gaussian_1, image_arr, kernel_size, sigma1, sigma2, threshold)
# filename = f"result/galata/edgeDetection/threshold_dog_convolve_{kernel_size}_{sigma1}_{sigma2}_{threshold}.png"
# save_binary_image(threshold_img_convolve, filename)


sigma1, sigma2, threshold = (1, 1, 0), (1.1, 1.1, 0), 254
threshold_img_gauss = ImageProcessor.threshold_dog_gaussian(smoothed_image_gaussian_1, sigma1, sigma2, threshold)
filename = f"result/galata/edgeDetection/threshold_dog_gaussian_{sigma1}_{sigma2}_{threshold}.png"
save_binary_image(threshold_img_gauss, filename)

# sigma1, sigma2, threshold = (1,1,0),(2,2,0),254
# threshold_img_gauss = ImageProcessor.threshold_dog_gaussian(smoothed_image_gaussian_1, sigma1, sigma2, threshold)
# filename = f"result/galata/edgeDetection/threshold_dog_gaussian_{sigma1}_{sigma2}_{threshold}.png"
# save_binary_image(threshold_img_gauss, filename)


# We continue with the threshold_dog_gaussian_(1, 1, 0)_(1.1, 1.1, 0)_254.png image that we like.

# -------------------------------------------------------------------------------------------------
# ------------------------------ Image Quantization Part -----------------------------------------
# -------------------------------------------------------------------------------------------------


quantization_hsv_image = ImageProcessor.quantization_hsv(smoothed_image_gaussian_1, 0.85)
save_image_3d_array(quantization_hsv_image, "result/galata/quantization/quantization_hsv_image_85.png")

# quantization_hsv_image = ImageProcessor.quantization_hsv(smoothed_image_gaussian_1, 0.35)
# save_image_3d_array(quantization_hsv_image, "result/galata/quantization/quantization_hsv_image_35.png")

# quantization_lab_image = ImageProcessor.quantization_lab(smoothed_image_gaussian_1, 0.85)
# save_image_3d_array(quantization_lab_image, "result/galata/quantization/quantization_lab_image_80.png")

# quantization_lab_image = ImageProcessor.quantization_lab(smoothed_image_gaussian_1, 0.35)
# save_image_3d_array(quantization_lab_image, "result/galata/quantization/quantization_lab_image_35.png")

quantization_KMeans_image = ImageProcessor.quantization_KMeans(smoothed_image_gaussian_1, num_colors=16)
save_image_3d_array(quantization_KMeans_image, "result/galata/quantization/quantization_KMeans_image_16.png")

# quantization_KMeans_image = ImageProcessor.quantization_KMeans(smoothed_image_gaussian_1, num_colors=4)
# save_image_3d_array(quantization_KMeans_image, "result/galata/quantization/quantization_KMeans_image_4.png")

# We continue with the
# result/galata/quantization/quantization_hsv_image_85.png ,
# result/galata/quantization/quantization_KMeans_image_16.png
# image that we like.

# -------------------------------------------------------------------------------------------------
# ------------------------------ Combining Edge and Quantized Image ------------------------------
# -------------------------------------------------------------------------------------------------


result_img_KMeans = quantization_KMeans_image * (1 - threshold_img_gauss).astype("uint8")
save_image_3d_array(result_img_KMeans, "result/galata/final/result_galata_img_KMeans.png")

result_img_hsv = quantization_hsv_image * (1 - threshold_img_gauss).astype("uint8")
save_image_3d_array(result_img_hsv, "result/galata/final/result_galata_img_hsv.png")
