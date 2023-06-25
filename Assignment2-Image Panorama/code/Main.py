import os 
import cv2
import numpy as np
import feature_extraction,feature_matching,find_homography,Merging_by_Transformation,extra
import glob
import time

# Get the list of folder names
folder_names = [os.path.basename(x) for x in glob.glob("dataset/" + "*")]



# Set the feature extraction method to be used
mthd = "ORB"

# Start the timer to calculate the elapsed time
start_time = time.time()

# Loop over all the folders in the dataset
for i in range(len(folder_names)):

    # Set the paths to the first and second images in the folder
    file_names =  [os.path.basename(x) for x in glob.glob("dataset/" + folder_names[i] + "/*")][5:11]
    file_names.sort()    
    first_img = f"dataset/{folder_names[i]}/{file_names[0]}"
    second_img = f"dataset/{folder_names[i]}/{file_names[1]}"
    img1 = cv2.imread(first_img)
    img2 = cv2.imread(second_img)

    # Resize the second image to match the size of the first image
    height, width = img1.shape[:2]
    img2 = cv2.resize(img2, (width, height))

     # Extract features from the first and second images using the specified method
    kp1, dp1, kp2, dp2 = feature_extraction.feature_extraction(img1, img2,method=mthd)
    # Find the matches between the features of the first and second images using the specified method
    matches_list,matches,src_pts,dst_pts = feature_matching.feature_matching(kp1, dp1, kp2, dp2,method=mthd)
     # Find the homography matrix between the first and second images
    H = find_homography.find_homography(src_pts, dst_pts)

    #Draw and save the feature matching image
    """extra_image = extra.drawMatches(img1,kp1,img2,kp2,matches)
    cv2.imwrite(f"{mthd}{folder_names[i]}extramerge1-2.png", extra_image)"""
    
    # Merge the first and second images using the homography matrix 
    output_img = Merging_by_Transformation.merge_image(img1,img2,src_pts,H)
    output_img = cv2.resize(output_img, (width, height))
    cv2.imwrite(f"{mthd}{folder_names[i]}merge1-2.png", output_img)

    #Repeat operations on other images
    for j in range(2,len(folder_names)):
        new_img_path = f"dataset/{folder_names[i]}/{file_names[j]}"
        new_img = cv2.imread(new_img_path)
        new_img = cv2.resize(new_img, (width, height))


        kp1, dp1, kp2, dp2 = feature_extraction.feature_extraction(output_img, new_img,method=mthd)
        matches_list,matches,src_pts,dst_pts = feature_matching.feature_matching(kp1, dp1, kp2, dp2,method=mthd)

        """extra_image = extra.drawMatches(output_img,kp1,new_img,kp2,matches)
        cv2.imwrite(f"{mthd}{folder_names[i]}extramerge1-{j+1}.png", extra_image)"""

        H = find_homography.find_homography(src_pts, dst_pts)
        output_img = Merging_by_Transformation.merge_image(output_img,new_img,src_pts,H)
        output_img = cv2.resize(output_img, (width, height))
        cv2.imwrite(f"{mthd}{folder_names[i]}merge1-{j+1}.png", output_img)



elapsed_time = time.time() - start_time
print(f"Elapsed time for the {mthd} method: ", elapsed_time)