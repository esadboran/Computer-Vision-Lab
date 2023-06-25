import cv2
import numpy as np

# Show keypoints on the image
def show_keypoints(img, kp):

    img_with_keypoints = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), \
                                           flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    
    return img_with_keypoints





#Feature matching result between two images 

def drawMatches(img,kp1,img2,kp2,matches):
    return  cv2.drawMatches(img, kp1, img2, kp2, matches, None, \
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

