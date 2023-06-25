import cv2
import numpy as np

def feature_matching(kp1, dp1, kp2, dp2, neighbors=2, ratio = 0.75 ,method='SIFT'):
    """
    Matches keypoints and descriptors between two images using the specified method.

    Args:
        kp1 (list): Keypoints for the first input image.
        dp1 (numpy.ndarray): Descriptors for the first input image.
        kp2 (list): Keypoints for the second input image.
        dp2 (numpy.ndarray): Descriptors for the second input image.
    Returns:
        Tuple[List[List[cv2.DMatch]], List[cv2.DMatch], numpy.ndarray, numpy.ndarray]: A tuple containing the matches and their corresponding source and destination keypoints.
    """

    # Create a matcher object based on the descriptor method
    if method in ['SIFT', 'SURF']:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    elif method == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        raise ValueError("Unsupported descriptor method.")
    
    # Match descriptors using k-NN
    matches_knn = matcher.knnMatch(dp1, dp2, k=neighbors)
    
    # Refine matches by applying the ratio test and incrementally increasing the ratio until the minimum number of matches is reached
    matches = list()
    while len(matches) < 10 and ratio < 0.8:
        ratio += 0.0125
        matches = list()
        for m, n in matches_knn:
            if m.distance < ratio * n.distance:
                matches.append(m)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Convert matches to the format expected by drawMatches
    matches_list = [[match] for match in matches]
    
    # Get source and destination keypoints for each match
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches_list])
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches_list])
    
    # Return the matches and their corresponding source and destination keypoints
    return matches_list, matches, src_pts, dst_pts