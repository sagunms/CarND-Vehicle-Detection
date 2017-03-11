import numpy as np
import cv2

# Class for perspective transform
class PerspectiveTransform():

    def __init__(self, img_size):

        # Define source image polygonal region of interest
        t_roi_y = np.uint(img_size[0] / 1.5)  # top y
        b_roi_y = np.uint(img_size[0])        # bottom y

        roi_x = np.uint(img_size[1] / 2)
        tl_roi_x = roi_x - 0.2 * np.uint(img_size[1] / 2) # top-left x
        tr_roi_x = roi_x + 0.2 * np.uint(img_size[1] / 2) # top-right x
        bl_roi_x = roi_x - 0.9 * np.uint(img_size[1] / 2) # bottom-left x
        br_roi_x = roi_x + 0.9 * np.uint(img_size[1] / 2) # bottom-right x

        # Define source image rectangle
        src = np.float32([[bl_roi_x, b_roi_y],
                          [br_roi_x, b_roi_y],
                          [tr_roi_x, t_roi_y],
                          [tl_roi_x, t_roi_y]])

        # Define destination image rectangle
        dst = np.float32([[0, img_size[0]],
                          [img_size[1], img_size[0]],
                          [img_size[1], 0],
                          [0, 0]])

        # Apply perspective transform
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
