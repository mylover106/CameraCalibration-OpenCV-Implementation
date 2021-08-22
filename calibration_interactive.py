#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import re


def cameraCalibrate(img_names, board_shape, visualize=False, visualize_shape=(4, 4)):
   
    # generate object points and calibrate camera
    bw, bh = board_shape
    x = np.arange(bw)
    y = np.arange(bh)
    yy, xx = np.meshgrid(y, x)
    obj_points = np.stack([xx, yy, np.zeros_like(xx)], axis=2)
    obj_points = obj_points.reshape(-1, 1, 3).astype(np.float32)

    w, h, _ = cv2.imread(img_names[0]).shape
    
    # get all image corner pts
    img_pts = list()
    obj_pts = list()
    plt.title('single camera cornert detection')
    # plt.figure(figsize=(20, 20))
    for i, img_name in enumerate(img_names):
        img_data = cv2.imread(img_name)
        flag, corner_pts = cv2.findChessboardCorners(img_data, (bh, bw))

        if i + 1 <= visualize_shape[0] * visualize_shape[1]:
            plt.subplot(visualize_shape[0], visualize_shape[1], i + 1)
            plt.imshow(img_data)
        
        if not flag: continue

        for i in range(9):
            plt.scatter(corner_pts[i * 5: (i + 1) * 5, 0, 0], corner_pts[i * 5: (i + 1) * 5, 0, 1], marker='^', linewidths=0.1)

        img_pts.append(corner_pts)
        obj_pts.append(obj_points)
    
    
    plt.show()
    
    ret_val, matrix, dist_coff, rvec, tvec = cv2.calibrateCamera(np.array(obj_pts), np.array(img_pts), (w, h), None, None)

    plt.title('single camera calibration')
    # plot undistorted images with detected points
    for i, img_name in enumerate(img_names):
        img_data = cv2.imread(img_name)

        new_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, dist_coff, (w, h), 0, (w, h))
    
        undistored_image = cv2.undistort(img_data, matrix, dist_coff, new_matrix)
        plt.subplot(visualize_shape[0], visualize_shape[1], i + 1)
        plt.imshow(undistored_image)
    
    
    plt.show()
    return ret_val, matrix, dist_coff, img_pts, obj_pts, w, h


def binocularCameraCalibrate(left_imgs, right_imgs, board_shape, visualize=False, visualize_shape=(4,4)):
    # check number consistency
    for left_name, right_name in zip(left_imgs, right_imgs):
        assert re.findall(r'd+', left_name) == re.findall(r'd+', right_name), f"{left_name}, {right_name} not consistent!"
    
    # calibrate single camera
    errleft, ml, dl, img_ptsl, obj_pts, w, h = cameraCalibrate(left_imgs, board_shape, visualize, visualize_shape)


    errright, mr, dr, img_ptsr, _, _, _ = cameraCalibrate(right_imgs, board_shape, visualize, visualize_shape)

    # calibrate binocular camera
    err, ml, dl, mr, dr, R, T, E, F = cv2.stereoCalibrate(obj_pts, img_ptsl, img_ptsr, ml, dl, mr, dr, (w, h), flags=cv2.CALIB_FIX_INTRINSIC)

    # stereo rectify
    Rl, Rr, Pl, Pr, Q, Roil, Roir = cv2.stereoRectify(ml, dl, mr, dr, (w, h), R, T)

    lmapx, lmapy = cv2.initUndistortRectifyMap(ml, dl, Rl, Pl[:,:3], (w, h), cv2.CV_32FC1)
    rmapx, rmapy = cv2.initUndistortRectifyMap(mr, dr, Rr, Pr[:,:3], (w, h), cv2.CV_32FC1)

    plt.figure(figsize=(10, 10), dpi=100)
    for i, img_pair in enumerate(zip(left_imgs, right_imgs)):
        left_img, right_img = cv2.imread(img_pair[0]), cv2.imread(img_pair[1])

        nleft = cv2.remap(left_img, lmapx, lmapy, cv2.INTER_LINEAR)
        nright = cv2.remap(right_img, rmapx, rmapy, cv2.INTER_LINEAR)

        # concat image
        full_img = cv2.hconcat([nleft, nright])
        h, w, _ = full_img.shape

        plt.subplot(visualize_shape[0], visualize_shape[1], i + 1)
        plt.imshow(full_img)

        # plot line
        for i in range(0, h, 50):
            plt.plot([0, w-1], [i, i], linewidth=0.5)
        
        

    plt.show()

    return ml, dl,mr, dr, R, T



if __name__ == '__main__':



    left_imgs = sorted(glob.glob('imgs/left/*.jpg'))[:4]
    right_imgs = sorted(glob.glob('imgs/right/*.jpg'))[:4]

    cam_ml, dist_l, cam_mr, dist_r, R, T = binocularCameraCalibrate(left_imgs, right_imgs, (7, 5), True,(2, 2))

