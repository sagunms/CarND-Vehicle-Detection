import numpy as np
import cv2
import math

from lane_lib import masking

# Class for lane detection on video frames
class LaneDetect():

    def __init__(self, M, Minv):

        self.M = M
        self.Minv = Minv
        self.initialised = False

        # BGR format
        self.colour_orange = (0, 165, 255) #(255, 0, 0)
        self.colour_red = (0, 0, 255)
        self.colour_green = (0, 255, 0)
        self.colour_light_blue = (180, 180, 50)

    # Main method to preprocess raw image, find lane lines and annotate input frame
    def detect(self, img, frame_overlay=True):

        # Apply binary masking
        self.mask = masking.BinaryMasking(img).mask
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = self.mask.shape

        # Warp image to bird-eye view perspective
        self.img_binary_warped = cv2.warpPerspective(self.mask, self.M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)
        self.img_colour_warped = cv2.warpPerspective(img, self.M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)

        # Initialise for the first time to calculate base lane positions
        # Otherwise, search around margin from previous position
        if not self.initialised:
            if not self.init():
                return self.result_img
        else:
            self.update()

         # Thickness of fit lane lines to be drawn
        margin = 30

        # Define search window around the previous left and right fits
        left_search1 = np.array([(np.vstack([self.left_polyfit_x - margin, self.plot_y])).T])
        left_search2 = np.array([np.flipud((np.vstack([self.left_polyfit_x + margin, self.plot_y])).T)])
        left_pts = np.hstack((left_search1, left_search2))
        right_search1 = np.array([(np.vstack([self.right_polyfit_x - margin, self.plot_y])).T])
        right_search2 = np.array([np.flipud((np.vstack([self.right_polyfit_x + margin, self.plot_y])).T)])
        right_pts = np.hstack((right_search1, right_search2))

        # Draw the lane onto the warped blank image
        img_search_window = np.zeros_like(self.img_colour_warped)
        
        cv2.fillPoly(img_search_window, np.int_([left_pts]), self.colour_red)
        cv2.fillPoly(img_search_window, np.int_([right_pts]), self.colour_green)

        left_line = np.array([(np.vstack([self.left_polyfit_x[::-1], self.plot_y[::-1]])).T])
        right_line = np.array([(np.vstack([self.right_polyfit_x, self.plot_y])).T])
        line_pts = np.hstack((left_line, right_line))
        cv2.fillPoly(img_search_window, np.int_([line_pts]), self.colour_light_blue)

        img_lane = cv2.warpPerspective(img_search_window, self.Minv, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)

        if frame_overlay:
            self.result_img = cv2.addWeighted(img, 1.0, img_lane, 0.6, 0)
        else:
            self.result_img = img_lane

        # Calculate curvature and annotate video
        self.lane_curvature()

        # Convert back to RGB colour space
        self.result_img = cv2.cvtColor(self.result_img, cv2.COLOR_BGR2RGB)

        return self.result_img

    # Calculate lane line curvature and annotate final image
    def lane_curvature(self):

        # Lane curvature
        y_m_px = 30. / 720      # meters/pixel in y direction
        x_m_px = 3.7 / 700      # meters/pixel in x direction

        # Calculate radius of curvature using maximum value of y (bottom-most of the frame)
        plot_y_max = np.max(self.plot_y)
        self.left_curve_rad = ((1 + (2 * self.left_polyfit[0] * plot_y_max * y_m_px + self.left_polyfit[1] * x_m_px)**2)**1.5) / np.absolute(2 * self.left_polyfit[0])
        self.right_curve_rad = ((1 + (2 * self.right_polyfit[0] * plot_y_max * y_m_px + self.right_polyfit[1] * x_m_px)**2)**1.5) / np.absolute(2 * self.right_polyfit[0])
        self.curvature = (self.left_curve_rad + self.right_curve_rad) / 2.0

        # Calculate offset of car from the centre
        self.offset = ((self.left_polyfit_x[10] + self.right_polyfit_x[10] - self.mask.shape[1]) / 2) * x_m_px

        if self.offset > 0:
            self.direction = 'Right'
        else:
            self.direction = 'Left'

        # Annotate video
        cv2.putText(self.result_img,
                    'Offset from Center = %.2f m (%s)' % (np.abs(self.offset), self.direction),
                    (int(self.mask.shape[1] * 0.35), 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.colour_orange, 2)

        cv2.putText(self.result_img,
                    'Radius of Curvature = %.2f m' % self.curvature,
                    (int(self.mask.shape[1] * 0.36), 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.colour_orange, 2)

    # Visualising Search Windows and Fit Lane Lines (used in Jupyter notebook)
    def draw_debug(self, img):

        # Undistort image frame
        img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        # Apply binary masking
        binary_masking = masking.BinaryMasking(img)
        self.mask = binary_masking.mask
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = self.mask.shape

        # Warp image to bird-eye view perspective
        self.img_binary_warped = cv2.warpPerspective(self.mask, self.M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)
        img_binary_warped2 = np.copy(self.img_binary_warped)

        # Draw the lane onto the warped blank image
        self.img_search_window = np.zeros_like(img)

        # Initialise for the first time to calculate base lane positions
        self.init(silent=False)

        # Thickness of fit lane lines to be drawn
        margin = 15

        # Define search window around the previous left and right fits
        left_search1 = np.array([(np.vstack([self.left_polyfit_x - margin, self.plot_y])).T])
        left_search2 = np.array([np.flipud((np.vstack([self.left_polyfit_x + margin, self.plot_y])).T)])
        left_pts = np.hstack((left_search1, left_search2))
        right_search1 = np.array([(np.vstack([self.right_polyfit_x - margin, self.plot_y])).T])
        right_search2 = np.array([np.flipud((np.vstack([self.right_polyfit_x + margin, self.plot_y])).T)])
        right_pts = np.hstack((right_search1, right_search2))

        # Draw Left and Right Lane Lines
        cv2.fillPoly(self.img_search_window, np.int_([left_pts]), self.colour_red)
        cv2.fillPoly(self.img_search_window, np.int_([right_pts]), self.colour_green)

        # Draw Lane surface area (Between Left and Right Lanes)
        img_lane_area = np.zeros_like(img)
        left_line = np.array([(np.vstack([self.left_polyfit_x[::-1], self.plot_y[::-1]])).T])
        right_line = np.array([(np.vstack([self.right_polyfit_x, self.plot_y])).T])
        line_pts = np.hstack((left_line, right_line))
        cv2.fillPoly(img_lane_area, np.int_([line_pts]), self.colour_light_blue)

        # Combination 1: Left and Right fit lane lines + Search Window
        img_binary_warped_rgb = cv2.cvtColor(binary_masking.cv2_uint8(self.img_binary_warped), cv2.COLOR_GRAY2BGR)
        result_img = cv2.addWeighted(img_binary_warped_rgb, 0.8, self.img_search_window, 1.0, 0)

        # Combination 2: Lane Area + Binary Mask
        img_binary_warped2 = cv2.cvtColor(binary_masking.cv2_uint8(img_binary_warped2), cv2.COLOR_GRAY2BGR)
        result_img = cv2.addWeighted(img_lane_area, 0.7, result_img, 1.0, 0)

        # Final combination: Combination 1 + Combination 2
        self.result_img = cv2.addWeighted(img_binary_warped2, 0.8, result_img, 1.0, 0)

    # Initialise lane lines base position using histogramming and windowing from
    def init(self, silent=True):

        # Take a histogram of the image
        histogram = np.sum(self.img_binary_warped[:,:], axis=0)

        # Find the peak of the left and right of the histogram as the initial x for the two lanes
        centre = np.int(histogram.shape[0] / 2)
        left_x_init = np.argmax(histogram[:centre])
        right_x_init = np.argmax(histogram[centre:]) + centre

        # Choose the number of vertically stacked windows to divide the frame into
        num_windows = 8

        # Set height of windows
        window_height = np.int(self.img_binary_warped.shape[0] / num_windows)

        # Identify the x and y positions of all nonzero pixels in the image
        valid_xy = self.img_binary_warped.nonzero()
        valid_y = np.array(valid_xy[0])
        valid_x = np.array(valid_xy[1])

        # Current positions to be updated for each window
        left_x_current = left_x_init
        right_x_current = right_x_init

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 30

        # Store left and right lane pixel indices
        left_lane_idx = []
        right_lane_idx = []

        # Iterate through the windows one by one
        for window in range(num_windows):
            # Find window boundaries in x and y covering both lane lines
            win_y_low = self.img_binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = self.img_binary_warped.shape[0] - window * window_height

            win_x_left_low = left_x_current - margin
            win_x_left_high = left_x_current + margin
            win_x_right_low = right_x_current - margin
            win_x_right_high = right_x_current + margin

            if not silent:

                # Draw windows and fit lane lines for visualisation
                left_rect_pts = [[win_x_left_low, win_y_low], 
                                 [win_x_left_low, win_y_high], 
                                 [win_x_left_high, win_y_high], 
                                 [win_x_left_high, win_y_low]]

                right_rect_pts = [[win_x_right_low, win_y_low], 
                                 [win_x_right_low, win_y_high], 
                                 [win_x_right_high, win_y_high], 
                                 [win_x_right_high, win_y_low]]

                cv2.fillPoly(self.img_binary_warped, np.int_([left_rect_pts]), self.colour_light_blue[::-1])
                cv2.fillPoly(self.img_binary_warped, np.int_([right_rect_pts]), self.colour_light_blue[::-1])

                cv2.rectangle(self.img_binary_warped, 
                              (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), 
                              thickness=3, color=self.colour_light_blue)
                cv2.rectangle(self.img_binary_warped, 
                              (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), 
                              thickness=3, color=self.colour_light_blue)

            good_left_idx_tmp = ((valid_y >= win_y_low) & 
                                 (valid_y < win_y_high) &
                                 (valid_x >= win_x_left_low) & 
                                 (valid_x < win_x_left_high)).nonzero()

            # Get the last nonzero index of left lane
            good_left_idx = good_left_idx_tmp[len(good_left_idx_tmp) - 1]

            good_right_idx_tmp = ((valid_y >= win_y_low) & 
                                  (valid_y < win_y_high) &
                                  (valid_x >= win_x_right_low) & 
                                  (valid_x < win_x_right_high)).nonzero()
            # Get the first nonzero index of right lane
            good_right_idx = good_right_idx_tmp[0]

            # Append these indices to the lists
            left_lane_idx.append(good_left_idx)
            right_lane_idx.append(good_right_idx)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_idx) > minpix:
                left_x_current = np.int(np.mean(valid_x[good_left_idx]))

            if len(good_right_idx) > minpix:
                right_x_current = np.int(np.mean(valid_x[good_right_idx]))

        # Concatenate the arrays of indices
        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)

        # Extract left and right line pixel positions
        left_x = valid_x[left_lane_idx]
        left_y = valid_y[left_lane_idx]
        right_x = valid_x[right_lane_idx]
        right_y = valid_y[right_lane_idx]

        # If empty pixels in any side, reject this frame
        if not (len(left_x) and len(left_y) and len(right_x) and len(right_y)):
            # import pdb;pdb.set_trace()
            self.initialised = False
            return False

        # Fit a 2nd order polynomial for left and right lane lines
        left_polyfit = np.polyfit(left_y, left_x, 2)
        right_polyfit = np.polyfit(right_y, right_x, 2)

        # Prepare x, y values for plotting
        plot_y = np.linspace(0, self.img_binary_warped.shape[0]-1, self.img_binary_warped.shape[0])
        left_polyfit_x = left_polyfit[0]*plot_y**2 + left_polyfit[1]*plot_y + left_polyfit[2]
        right_polyfit_x = right_polyfit[0]*plot_y**2 + right_polyfit[1]*plot_y + right_polyfit[2]

        self.left_lane_idx = left_lane_idx
        self.right_lane_idx = right_lane_idx
        self.plot_y = plot_y
        self.left_polyfit = left_polyfit
        self.right_polyfit = right_polyfit
        self.left_polyfit_x = left_polyfit_x
        self.right_polyfit_x = right_polyfit_x

        self.initialised = True

        return True

    # Update search around margin from previous position
    def update(self):

        img_binary_warped = self.img_binary_warped
        valid_xy = img_binary_warped.nonzero()
        valid_y = np.array(valid_xy[0])
        valid_x = np.array(valid_xy[1])
        margin = 30

        self.left_lane_idx = ((valid_x > (self.left_polyfit[0] * (valid_y**2) + self.left_polyfit[1] * valid_y + self.left_polyfit[2] - margin)) &
                              (valid_x < (self.left_polyfit[0] * (valid_y**2) + self.left_polyfit[1] * valid_y + self.left_polyfit[2] + margin)))
        self.right_lane_idx = ((valid_x > (self.right_polyfit[0] * (valid_y**2) + self.right_polyfit[1] * valid_y + self.right_polyfit[2] - margin)) &
                               (valid_x < (self.right_polyfit[0] * (valid_y**2) + self.right_polyfit[1] * valid_y + self.right_polyfit[2] + margin)))

        # Get pixel positions of both lane sides
        left_x = valid_x[self.left_lane_idx]
        left_y = valid_y[self.left_lane_idx]
        right_x = valid_x[self.right_lane_idx]
        right_y = valid_y[self.right_lane_idx]

        # import pdb;pdb.set_trace()
        if not (len(left_x) and len(left_y) and len(right_x) and len(right_y)):
            self.initialised = False
            return False

        # Fit 2nd order polynomial to left and right lines
        self.left_polyfit = np.polyfit(left_y, left_x, 2)
        self.right_polyfit = np.polyfit(right_y, right_x, 2)

        # Prepare x, y values for plotting
        self.plot_y = np.linspace(0, img_binary_warped.shape[0]-1, img_binary_warped.shape[0])
        self.left_polyfit_x = self.left_polyfit[0]*self.plot_y**2 + self.left_polyfit[1]*self.plot_y + self.left_polyfit[2]
        self.right_polyfit_x = self.right_polyfit[0]*self.plot_y**2 + self.right_polyfit[1]*self.plot_y + self.right_polyfit[2]

        return True
