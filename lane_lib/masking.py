import numpy as np
import cv2

from vehicle_lib.config import *

# Class for setting up binary mask for images
class BinaryMasking():

    def __init__(self, img):
        # Colour masking
        img_masked_y_channel = self.colour_filter_y_channel(img, config['colour_filter_y'])
        img_masked_v_channel = self.colour_filter_v_channel(img, config['colour_filter_v'])

        # Sobel edge masking
        img_abs_x = self.sobel_gradient_xy(img, config['sobel_grad_xy'], config['sobel_grad_xy_th'])
        img_g_mag = self.sobel_gradient_magnitude(img, config['sobel_grad_mag'], config['sobel_grad_mag_th'])
        img_d_mag = self.sobel_gradient_direction(img, config['sobel_grad_dir'], config['sobel_grad_dir_th'])

        # Combined masking
        mask = np.zeros_like(img_d_mag)
        mask[((img_masked_y_channel == 1) | (img_masked_v_channel == 1)) & (img_abs_x == 1) | 
              ((img_g_mag == 1) & (img_d_mag == 1))] = 1

        self.mask = mask

    def cv2_uint8(self, img):
        return cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Filter Y channel lane colour
    def colour_filter_y_channel(self, img, threshold=(0, 255)):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_y = img_yuv[:, :, 0]
        img_binary = np.zeros_like(img_y)
        img_binary[(img_y > threshold[0]) & (img_y <= threshold[1])] = 1

        return img_binary

    # Filter V channel lane colour
    def colour_filter_v_channel(self, img, threshold=(0, 255)):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_v = img_yuv[:, :, 2]
        img_binary = np.zeros_like(img_v)
        img_binary[(img_v > threshold[0]) & (img_v <= threshold[1])] = 1
      
        return img_binary

    # Calculate directional gradient
    def sobel_gradient_xy(self, img, orient='x', threshold=(0, 255)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            img_s = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        else:
            img_s = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        img_abs = np.absolute(img_s)
        img_sobel = np.uint8(255 * img_abs / np.max(img_abs))

        img_binary = np.zeros_like(img_sobel)
        img_binary[(img_sobel >= threshold[0]) & (img_sobel <= threshold[1])] = 1

        return img_binary

    # Calculate gradient magnitude
    def sobel_gradient_magnitude(self, img, kernel=3, threshold=(0, 255)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img_sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
        img_sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)

        img_s = np.sqrt(img_sx**2 + img_sy**2)
        img_s = np.uint8(255 * img_s / np.max(img_s))

        img_binary = np.zeros_like(img_s)
        img_binary[(img_s >= threshold[0]) & (img_s <= threshold[1])] = 1

        return img_binary

    # Calculate gradient direction
    def sobel_gradient_direction(self, img, kernel=3, threshold=(0, np.pi/2)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
        img_sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
        grad_s = np.arctan2(np.absolute(img_sy), np.absolute(img_sx))
        img_binary = np.zeros_like(grad_s)
        img_binary[(grad_s >= threshold[0]) & (grad_s <= threshold[1])] = 1

        return img_binary
