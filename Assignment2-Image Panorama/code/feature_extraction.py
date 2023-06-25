import cv2

def feature_extraction(img1, img2, method: str = "SIFT"):
    """
    Extracts features from two images using the specified method.

    Args:
        img1 (numpy.ndarray): The first input image.
        img2 (numpy.ndarray): The second input image.
        method (str, optional): The feature extraction method to use. Valid options are "SIFT", "SURF", and "ORB". Defaults to "SIFT".

    Returns:
        Tuple[List[cv2.KeyPoint], numpy.ndarray, List[cv2.KeyPoint], numpy.ndarray]: A tuple containing the keypoints and descriptors for both input images.
    """

    # Determine the feature detector based on the selected method
    feature_detector = cv2.SIFT_create() if method == "SIFT" else \
                       cv2.xfeatures2d.SURF_create() if method == "SURF" else \
                       cv2.ORB_create() # cv2.SURF_create() doesn't work

    # Detect keypoints and compute descriptors for both images
    kp1, dp1 = feature_detector.detectAndCompute(img1, None)
    kp2, dp2 = feature_detector.detectAndCompute(img2, None)

    # Return the keypoints and descriptors for both images
    return kp1, dp1, kp2, dp2