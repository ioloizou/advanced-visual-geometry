"""
AVG Lab 2 - Part 2 : Epipolar Geometry, Computing fundamental matrix
Example use:
> cd lab2_folder
> python lab2_part2.py 

Submission: Please submit this code completed and commented. You need to comment
everything you add and explain why you add it.
"""

import argparse
from matplotlib import pyplot as plt
import numpy as np
import cv2
from constants import F_gt
# The following imports are for printing with file name and line numbers
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

def normalize_transformation(points: np.ndarray) -> np.ndarray:
    """
    Compute a similarity transformation matrix that translate the points such that
    their center is at the origin & the avg distance from the origin is sqrt(2)
    :param points: <float: num_points, 2> set of key points on an image
    :return: (sim_trans <float, 3, 3>)
    """

    # Find center of the set of points by computing mean of x & y
    center = np.mean(points, axis=0) 
    
    # Matrix of distance from every point to the origin, shape: <num_points, 1>
    dist = np.linalg.norm(points - center, axis=1).reshape(-1, 1) 
    
    # Scale factor the similarity transformation = sqrt(2) / (mean of dist)
    s = np.sqrt(2) / np.mean(dist) 
    sim_trans = np.array([
        [s,     0,      -s * center[0]],
        [0,     s,      -s * center[1]],
        [0,     0,      1]
    ])
    return sim_trans

def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinate
    :param points: <float: num_points, num_dim>
    :return: <float: num_points, 3>
    """
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

def main(args):
    """
    In this part, we will estimate the fundamental matrix F from point correspondences
    between two images using the normalized 8-point algorithm.

    Input:
    - Two images of the same scene taken from different viewpoints

    Output:
    - The estimated fundamental matrix F
    """
    logging.info("args = %s", args)

    # Read image & put them in grayscale
    # queryImage
    img1 = cv2.imread(args.img_1, 0)
    # trainImage   
    img2 = cv2.imread(args.img_2, 0)  
    

    # The following part is for automatically getting pairs of corresponding points from two images
    orb = cv2.ORB_create()
    # Detect keypoints & compute descriptors automatically (Don't hesitate to google/LLM what this does)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match keypoints using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)


    # Organize key points into matrix, each row is a point
    query_kpts = np.array([kp1[m.queryIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>
    train_kpts = np.array([kp2[m.trainIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>

    # plot matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_matches)
    plt.title('Matched Keypoints')
    plt.show()

    # The following follows Algorithm 1 from the lab subject
    # Normalize keypoints
    T_query = normalize_transformation(query_kpts)  # get the similarity transformation for normalizing query kpts
    T_train = normalize_transformation(train_kpts)  # get the similarity transformation for normalizing train kpts

    # Homogenize and transform keypoints
    # ... you may need to add more lines here
    
    # Homogenize the points
    normalized_query_kpts = homogenize(query_kpts) 
    normalized_train_kpts = homogenize(train_kpts)

    # Apply the transformation to the points
    normalized_query_kpts = np.dot(T_query, normalized_query_kpts.T).T
    normalized_train_kpts = np.dot(T_train, normalized_train_kpts.T).T

    # Construct homogeneous linear equation to find fundamental matrix
    # ... you may add lines here
    
    # Constucting A matrix for the linear equation
    A = np.zeros((len(normalized_query_kpts), 9))
    for i in range(len(normalized_query_kpts)):
        x, y, _ = normalized_query_kpts[i]
        u, v, _ = normalized_train_kpts[i]
        A[i] = [u*x, u*y, u, v*x, v*y, v, x, y, 1]

    # Find vector f by solving A f = 0 using SVD
    # ... you may add lines here
    f = np.linalg.svd(A)[2][-1]


    # Arrange f into 3x3 matrix to get fundamental matrix F
    F = f.reshape(3, 3)
    print('rank F: ', np.linalg.matrix_rank(F))  # should be = 3

    # Force F to have rank 2
    # ... you may need to add more lines here
    # Approximate F with the close matrix F_hat under Frobenius norm.
    # Do this keeping only the largest singular value of F and setting the others to 0
    U, S, V = np.linalg.svd(F)
    print('S: ', S)
    S[-1] = 0
    print('S: ', S)
    F = np.dot(U, np.dot(np.diag(S), V))
    assert np.linalg.matrix_rank(F) == 2, 'Fundamental matrix must have rank 2'

    # De-normalize F
    # hint: last line of Algorithme 1 in the lab subject
    F = np.dot(T_train.T, np.dot(F, T_query))

    # Check if F is correct
    logging.info("(F - F_gt) = %s", F - F_gt)
    # Save new F in order to test using lab2_part1.py
    np.savetxt(args.F_new, F)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--img_1', help='', default="img/chapel_0.png")
    parser.add_argument('--img_2', help='', default="img/chapel_1.png")
    parser.add_argument('--F_new', help='', default="estimated_F.txt")
    args = parser.parse_args()

    main(args)


