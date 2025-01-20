"""
    Skeleton code for lab 1
    Usage : python path/to/lab1_skeleton.py --filename path/to/image

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse
# The following import is to enable logging with line number (instead of printing)
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

# Utility functions, you can mostly ignore them
def euclidean_trans(theta, tx, ty):
    return np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta), np.cos(theta), ty],
        [0, 0, 1]
    ])

def points_that_should_work_if_pointing_does_not_work():
    return np.array([
        [ 745.65367965,  646.25324675],
        [1057.34199134,  464.43506494],
        [1382.01731602,  641.92424242],
        [1061.67099567,  867.03246753]
    ])

def get_bounds(points):
    """Calculate the bounds of transformed points"""
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    return min_x, max_x, min_y, max_y

def auto_euclidean_trans(points, img_shape):
    """Automatically compute translation to center the transformed points"""
    min_x, max_x, min_y, max_y = get_bounds(points)
    
    # Calculate center of transformed points
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    
    # Calculate translation to center of image
    tx = img_shape[1]/2 - center_x
    ty = img_shape[0]/2 - center_y
    
    return euclidean_trans(np.deg2rad(0), tx, ty)

def plot_lines_and_vanishing_point(img, line1, line2, vanishing_point, 
                                   title="lines and vanishing point", 
                                   line1_label="Line 1", 
                                   line2_label="Line 2"):
    """
    Plot two lines and their vanishing point on the image
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    x_vals = np.array(ax.get_xlim())
    # Plot lines
    y_vals_1 = -(line1[0] * x_vals + line1[2]) / line1[1]
    y_vals_2 = -(line2[0] * x_vals + line2[2]) / line2[1]
    ax.plot(x_vals, y_vals_1, '--r', label=line1_label)
    ax.plot(x_vals, y_vals_2, '--b', label=line2_label)
    # Plot vanishing point
    ax.plot(vanishing_point[0], vanishing_point[1], 'ro', label='Vanishing point')
    ax.legend()
    plt.title(title)
    plt.show()

# You can use this function to get points from the image
def acquire_points(image, n=4):
    fig, ax = plt.subplots()
    ax.imshow(image)
    points = plt.ginput(n)# TODO
    plt.close(fig)  # Close the figure after getting points
    return np.array(points)

def main(args): # args come from the script at the end of the file
    """
    Goal : Perform affine rectification then metric rectification on the given image
    """
    # Read image
    img = cv2.imread(args.filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    logging.info('\n-------- Task 1: Affine rectification --------')
    # Get 2 pairs of parallel lines in the affine image
    # (You can use the following incomplete function to get points)
    points = acquire_points(img, n=4)
    logging.info(f'chosen points: {points}')  # each row is a point
    
    # [Tip : uncomment the code step by step once the previous step is working]
    # convert chosen pts to homogeneous coordinate
    pts_homo = np.concatenate((points, np.ones((4, 1))), axis=1)#TODO
    
    # # Identify image of the line at infinity on the projective plane
    line_0_0 = np.cross(pts_homo[0], pts_homo[1]) #TODO
    line_0_1 = np.cross(pts_homo[2], pts_homo[3]) #TODO
    pt_vanishing_0 = np.cross(line_0_0, line_0_1) #TODO # first vanishing point
    pt_vanishing_0 /= pt_vanishing_0[-1]  # normalize
    logging.info('First vanishing point: %s', pt_vanishing_0)
    
    # Debug : Plot first pair of lines and vanishing point
    plot_lines_and_vanishing_point(img, line_0_0, line_0_1, pt_vanishing_0)

    line_1_0 = np.cross(pts_homo[0], pts_homo[2])#TODO
    line_1_1 = np.cross(pts_homo[1], pts_homo[3])#TODO
    pt_vanishing_1 = np.cross(line_1_0, line_1_1)#TODO # 2nd vanishing point

    pt_vanishing_1 /= pt_vanishing_1[-1]
    logging.info('Second vanishing point: %s', pt_vanishing_1)

    # Debug : Plot second pair of lines lines and vanishing point
    plot_lines_and_vanishing_point(img, line_1_0, line_1_1, pt_vanishing_1)

    l_inf = np.cross(pt_vanishing_0, pt_vanishing_1)#TODO # image of line at inf
    l_inf /= l_inf[-1]
    logging.info('Line at infinity: %s', l_inf)

    # Construct the projectivity that affinely rectify image
    H = np.array([
                 [1, 0, 0],
                 [0, 1, 0],
                 [l_inf[0], l_inf[1], l_inf[2]]
                 ])#TODO 
    
    # Check your results : what should be the image of line at inf on affinely rectified image?
    logging.info('Image of line at inf on affinely rectified image: %s', 
                 (np.linalg.inv(H).T @ l_inf.reshape(-1, 1)).squeeze())

    # H_E is a Euclidean transformation to center the image for visualization
    H_E = euclidean_trans(np.deg2rad(0), 50, 250)
    view_H = H_E @ H
    affine_img = cv2.warpPerspective(img, view_H, (img.shape[1], img.shape[0]))#TODO 
    affine_pts = (view_H @ pts_homo.T).T#TODO 
    for i in range(affine_pts.shape[0]):
        affine_pts[i] /= affine_pts[i, -1]

    plt.plot(*zip(*affine_pts[:, :-1]), marker='o', color='r', ls='')
    plt.imshow(affine_img)
    plt.show()

    logging.info('\n-------- Task 2: Metric rectification --------')
    # Get 2 pairs of orthogonal lines in the affine image
    # (you can re-use the points from the parallel lines))
    aff_line_0_0 = np.cross(affine_pts[0], affine_pts[1])#TODO
    aff_line_0_1 = np.cross(affine_pts[2], affine_pts[3])#TODO

    aff_line_1_0 = np.cross(affine_pts[0], affine_pts[2])#TODO
    aff_line_1_1 = np.cross(affine_pts[1], affine_pts[3])#TODO

    aff_line_0_0 /= aff_line_0_0[-1]
    aff_line_0_1 /= aff_line_0_1[-1]
    aff_line_1_0 /= aff_line_1_0[-1]
    aff_line_1_1 /= aff_line_1_1[-1]

    # Construct constraint matrix C to find vector s
    C0 = np.array([aff_line_0_0[0]*aff_line_1_0[0], aff_line_0_0[0]*aff_line_1_0[1] + aff_line_0_0[1]*aff_line_1_0[0], aff_line_0_0[1]*aff_line_1_0[1]])#TODO 
    C1 = np.array([aff_line_0_1[0]*aff_line_1_1[0], aff_line_0_1[0]*aff_line_1_1[1] + aff_line_0_1[1]*aff_line_1_1[0], aff_line_0_1[1]*aff_line_1_1[1]])#TODO 
    C = np.vstack([C0, C1])
    logging.info('Constraint matrix C:\n%s', C)


    # Find s by looking for the kernel of C (hint: SVD)
    _, _, Vt = np.linalg.svd(C)
    s = Vt[-1, :]#TODO 

    mat_S = np.array([
        [s[0], s[1]],
        [s[1], s[2]],
    ])
    logging.info('Matrix S:\n%s', mat_S)

    # Find the projectivity that do metric rectification
    # if #TODO:
    #     raise ValueError("Error: Found non-positive eigenvalues. The matrix S should be positive definite.")
    eigenvalues, eigenvectors = np.linalg.eig(mat_S)
    print(eigenvalues)
    sqrt_eigenvalues = np.diag(np.sqrt(eigenvalues))
    K = eigenvectors @ sqrt_eigenvalues
    H = np.zeros((3, 3))
    H[:2, :2] = np.linalg.inv(K)
    H[2, 2] = 1#TODO
    
    # Check results by computing dual conic, what should be the image 
    # of dual conic on the metric rectified image?
    aff_dual_conic = np.array([
        [s[0], s[1], 0],
        [s[1], s[2], 0],
        [0, 0, 0]
    ])
    logging.info('Image of dual conic on metric rectified image: %s', H @ aff_dual_conic @ H.T)

    # Automatically find H_E for centering image
    temp_pts = (H @ affine_pts.T).T
    for i in range(temp_pts.shape[0]):
        temp_pts[i] /= temp_pts[i, -1]
    H_E = auto_euclidean_trans(temp_pts, img.shape)
   
    view_H = H_E @ H

    # Warp image and points
    eucl_img = cv2.warpPerspective(affine_img, view_H, (img.shape[1], img.shape[0])) 
    eucl_pts = (view_H @ affine_pts.T).T#TODO
    for i in range(eucl_pts.shape[0]):
        eucl_pts[i] /= eucl_pts[i, -1]
    plt.plot(*zip(*eucl_pts[:, :-1]), marker='o', color='r', ls='')
    plt.imshow(eucl_img)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    args = parser.parse_args()
    main(args)