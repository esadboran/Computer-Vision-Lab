import cv2 
import numpy as np


def find_homography(src_pts, dst_pts, ransac_thresh=5, ransac_max_iter=1000):
    """
    Computes homography matrix using RANSAC algorithm.

    Args:
        src_pts: Source points (Nx2 numpy array).
        dst_pts: Destination points (Nx2 numpy array).
        ransac_thresh: RANSAC threshold (default: 5).
        ransac_max_iter: Maximum iterations for RANSAC (default: 1000).

    Returns:
        Homography matrix (3x3 numpy array).
    """    

    # Initialize variables to store best homography matrix and number of inliers
    best_H = None
    best_num_inliers = 0
    
    # Perform RANSAC iterations
    for i in range(ransac_max_iter):
        
        # Select random 4 points from the source and destination arrays
        idx = np.random.choice(src_pts.shape[0], 4, replace=False)
        src = src_pts[idx]
        dst = dst_pts[idx]

        # P matrixi oluşturdum
        num_points = src.shape[0]
        P = np.zeros((2*num_points+1,9))
        for i in range(num_points):
            x, y, z = src[i][0], src[i][1], 1
            x_t, y_t, z_t = dst[i][0], dst[i][1], 1
            
            P[2*i,:] = np.array([0, 0, 0, -z_t*x, -z_t*y, -z_t*z, y_t*x, y_t*y, y_t*z])
            P[2*i+1,:] = np.array([z_t*x, z_t*y, z_t*z, 0, 0, 0, -x_t*x, -x_t*y, -x_t*z])

        P[-1]= [0, 0, 0, 0, 0, 0, 0, 0, 1]

        if np.linalg.det(P) == 0:
            continue

        # Calculate homography matrix using least squares method
        unit_column_vector = np.zeros((9, 1))
        unit_column_vector[8] = 1
        p_inv = np.linalg.inv(P)
        H_matrix = np.dot(p_inv, unit_column_vector)
        H_matrix = np.reshape(H_matrix, (3, 3))

        # Calculate number of inliers using the homography matrix and threshold value
        num_inliers = 0
        for j in range(src_pts.shape[0]):
            src_pt = np.append(src_pts[j], 1)
            dst_pt = np.append(dst_pts[j], 1)
            dst_pt_pred = np.dot(H_matrix, src_pt)
            dst_pt_pred /= dst_pt_pred[-1]
            if np.linalg.norm(dst_pt - dst_pt_pred) < ransac_thresh:
                num_inliers += 1

        # Update best homography matrix and number of inliers if the current iteration yields better results
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_H = H_matrix

    #Round the numbers in H matrix
    np.set_printoptions(precision=5, suppress=True)
    return best_H