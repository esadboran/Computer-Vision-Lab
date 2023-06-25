import numpy as np
import cv2


def merge_image(img_one, img_two,src_pts,H):
    
    # Get the x and y coordinates of the first feature point
    x = src_pts[0][0].astype(int)
    y = src_pts[0][1].astype(int)

    # Transform the coordinates using homography matrix H
    prime = [[x], [y], [1]]
    prime_inverse = np.dot(H, prime)
    
    # Get the x-coordinate of the transformed point
    x_merge = (prime_inverse[0] / prime_inverse[2])[0].astype(int)

    # Crop the left part of img_one and the right part of img_two
    img_one = img_one[:, 0:x]
    img_two = img_two[:, x_merge:]

    # Concatenate the two images horizontally to create a panoramic image
    panoramic_img = np.hstack((img_one, img_two))

    return panoramic_img