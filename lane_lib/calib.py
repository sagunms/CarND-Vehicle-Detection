import numpy as np
import cv2
import glob
import pickle as pickle
import os

class CameraCalibrate():

    def __init__(self,
                 corners_shape = (9, 6),
                 img_glob = "camera_cal/calibration*.jpg",
                 calib_fname = "calib.p"):

        if os.path.exists(calib_fname):
            # If calibration file already exists, just get mtx, dist from it.
            with open(calib_fname, 'rb') as f:
                calib_data = pickle.load(f)
                mtx = calib_data["cam_matrix"]
                dist = calib_data["dist_coeffs"]
        else:
            # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((corners_shape[0] * corners_shape[1], 3), np.float32)
            objp[:,:2] = np.mgrid[0: corners_shape[0], 
                                  0: corners_shape[1]].T.reshape(-1, 2) # x, y coordinates

            # Arrays to store object points and image points from all the images.
            obj_points = []     # 3d points in real world space
            img_points = []     # 2d points in image plane.

            # Make a list of calibration images
            images = glob.glob(img_glob)

            # step through the list and search for chessboard corners
            for fname in images:
                # Read image
                img = cv2.imread(fname)

                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, corners_shape, None)

                # If found, add object points, image points
                if ret == True:
                    obj_points.append(objp)
                    img_points.append(corners)

            # Test undistortion on an image
            img_size = (img.shape[1], img.shape[0])

            # Calibrate camera given the object and image points
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, 
                                                               img_size, None, None)

            # Pickle the data and save it
            calib_data = {'cam_matrix': mtx,
                          'dist_coeffs': dist,
                          'img_size': img.shape}

            with open(calib_fname, 'wb') as f:
                pickle.dump(calib_data, f)

        self.mtx = mtx
        self.dist = dist

    def undistort(self, img):
        # Undistort image frame
        img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return img

    def draw(self):
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        cv2.imshow('img', img)