"""
AVG Lab 2 - Part 1 : Epipolar Geometry, Using Fundamental Matrix
Example use:
> cd lab2_folder
> python lab2_part1.py

Submission: Please submit this code completed and commented. You need to comment
everything you add and explain why you add it.
"""

import os
import shutil
import argparse
import logging
from matplotlib import pyplot as plt
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import cv2
import numpy as np

def get_points(image, n=1):
    fig, ax = plt.subplots()
    ax.imshow(image)
    points = plt.ginput(n)
    plt.close(fig)  # Close the figure after getting points
    return np.array(points)

def main(args):
    """
    In this part, we will compute the epipolar lines and the epipole given the F matrix 
    and plot them on the second image.

    Input: 
    - chapel_0.png: the first image
    - chapel_1.png: the second image
    - F_gt: the ground truth fundamental matrix
    - User input for the points on the first image 
     (you should prompt the user to input the points !!!)

    Output: a single plot showing the chapel_1.png image with the epipolar lines 
    and the epipole (a dot).
    """
    from constants import F_gt as F
    logging.info("args = %s", args)

    # # Load estimated_F.txt and store it in F
    # if os.path.exists('estimated_F.txt'):
    #     F = np.loadtxt('estimated_F.txt')
    #     logging.info('F = %s', F)

    # Load the images
    img_0 = cv2.imread(args.img_0)
    img_1 = cv2.imread(args.img_1)

    # Check if images are loaded correctly
    if img_0 is None:
        raise FileNotFoundError(f"Image {args.img_0} not found.")
    if img_1 is None:
        raise FileNotFoundError(f"Image {args.img_1} not found.")

    # Get the points from the user
    points = get_points(img_0, n=3)

    # Compute the epipolar lines
    lines = cv2.computeCorrespondEpilines(points, 1, F)

    # Plot the epipolar lines
    fig, ax = plt.subplots()
    ax.imshow(img_1)

    # Iterate over each epipolar line
    for i, line in enumerate(lines):
        a, b, c = line[0]  # Extract the coefficients of the line equation ax + by + c = 0
        x0, x1 = 0, img_1.shape[1]  # Define the x-coordinates for the line endpoints
        y0 = -c / b  # Compute the y-coordinate for the left endpoint (x0)
        y1 = -(a * x1 + c) / b  # Compute the y-coordinate for the right endpoint (x1)
        ax.plot([x0, x1], [y0, y1], label=f"Line {i+1}")  # Plot the line on the second image

    # Compute the epipole point 
    U, S, V = np.linalg.svd(F)
    e = V[-1] / V[-1, -1]  # Normalize the last column of V
    ax.plot(e[0], e[1], 'ro', label='Epipole')  # Plot the epipole on the second image

    # Show the legend on the top right corner
    ax.legend(loc='upper right')
    
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--img_0', help='', default='chapel_0.png')
    parser.add_argument('--img_1', help='', default='chapel_1.png')
    args = parser.parse_args()

    main(args)