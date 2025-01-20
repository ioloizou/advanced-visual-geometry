"""
AVG Lab 2 - Part 3 : Stereo
Example use:
> cd lab2_folder
> python lab2_part3.py

Submission: Please submit this code completed and commented. You need to comment
everything you add and explain why you add it.
"""

import cv2
import open3d as o3d
import argparse
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
from constants import extrinsic, K, Bf

def main(args):
    """
    In this part, we will reconstruct a 3D point cloud from a pair of stereo images.

    Input:
    - left.png: the left stereo image (grayscale)
    - right.png: the right stereo image (grayscale) 
    - left_color.png: the left stereo image (color) - used for point cloud coloring
    - camera calibration parameters from constants.py (K, Bf)

    Output: 
    - Interactive 3D visualization of the reconstructed colored point cloud
    """
    logging.info("args = %s", args)

    # load images
    left = cv2.imread(args.left_img, 0)
    right = cv2.imread(args.right_img, 0)
    left_color = cv2.imread(args.left_color_img)

    # compute disparity
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
    disparity = stereo.compute(left, right)  # an image of the same size as "left" (or "right")

    # Create array of pixels with positive disparity (num_pixels_with_disparity, 2)
    # ... you may need to add more lines here
    pixels_with_disparity = np.argwhere(disparity > 0)
    logging.info('pixels_with_disparity.shape=%s',pixels_with_disparity.shape)
    
    # Stack the positive disparity with the coordinates
    # ... you may need to add more lines here
    pixels_disparities = disparity[pixels_with_disparity[:, 0], pixels_with_disparity[:, 1]]
    pixels_with_disparity = np.hstack((pixels_with_disparity, pixels_disparities[:, np.newaxis]))
    logging.info('pixels_with_disparity.shape=%s',pixels_with_disparity.shape)
    
    # Compute the depth of every pixel whose disparity is positive
    # hint: assume d is the disparity of pixel (u, v)
    # hint: the depth Z of this pixel is Z = Bf / d
    Z = Bf / pixels_with_disparity[:, 2] 
    
    # Compute normalized coordinates of every pixel whose disparity is positive
    # hint: the normalized coordinate of pixel [u, v, 1] is K^(-1) @ [u, v, 1]
    # ... you may need to add more lines here
    normalized_coordinates = np.linalg.inv(K) @ np.hstack((pixels_with_disparity[:, :2], np.ones((pixels_with_disparity.shape[0], 1)))).T
    logging.info('normalized_coordinates.shape=%s',normalized_coordinates.shape)

    # Compute 3D coordinate of every pixel whose disparity is positive
    # hint: 3D coordinate of pixel (u, v) is the product of Z and its normalized coordinate
    # ... you may need to add more lines here
    all_3d = Z[: , np.newaxis] * normalized_coordinates.T
    logging.info('all_3d.shape=%s',all_3d.shape)

    # Get color for 3D points
    all_color = left_color[pixels_with_disparity[:, 0], pixels_with_disparity[:, 1]]
    logging.info('all_color.shape=%s',all_color.shape)
    
    # Normalize all_color
    all_color = all_color.astype(float) / 255.0

    # The following code is for displaying the 3D pointcloud, you don't need to modify it
    # Display 3D pointcloud
    cloud = o3d.geometry.PointCloud()  # create pointcloud object
    cloud.points = o3d.utility.Vector3dVector(all_3d)
    cloud.colors = o3d.utility.Vector3dVector(all_color)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])  # create frame object
    # Create visualizer and set initial viewpoint
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)  # Can now use any window size
    vis.add_geometry(cloud)
    vis.add_geometry(mesh_frame)
    # Set initial camera viewpoint
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    # Set camera intrinsics from JSON
    params.intrinsic.set_intrinsics(
        width=1920,
        height=1043,
        fx=903.26449614716955,
        fy=903.26449614716955,
        cx=959.5,
        cy=521.0
    )
    params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--left_img', help='', default="img/left.png")
    parser.add_argument('--right_img', help='', default="img/right.png")
    parser.add_argument('--left_color_img', help='', default="img/left_color.png")
    args = parser.parse_args()

    main(args)

