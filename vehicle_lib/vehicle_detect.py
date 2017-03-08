import cv2
import numpy as np
from scipy.ndimage.measurements import label
from skimage.feature import hog

from vehicle_lib import feature_extract
from vehicle_lib import window
from vehicle_lib import heatmap

# Main class of the project.
# Encapsulates sliding windows, feature generation, svm, remove duplicates and false positives, etc.
class VehicleDetect:

    def __init__(self, color_space, orient, pix_per_cell, cell_per_block,
                 hog_channel, spatial_size, hist_bins, spatial_feat,
                 hist_feat, hog_feat, y_start_stop, x_start_stop, xy_window,
                 xy_overlap, heatmap_threshold, #scaler,
                 classifier):

        self.color_space = color_space          # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = orient                    # HOG orientations
        self.pix_per_cell = pix_per_cell        # HOG pixels per cell
        self.cell_per_block = cell_per_block    # HOG cells per block
        self.hog_channel = hog_channel          # Can be 0, 1, 2, or "ALL"
        self.spatial_size = spatial_size        # Spatial binning dimensions
        self.hist_bins = hist_bins              # Number of histogram bins
        self.spatial_feat = spatial_feat        # Spatial features on or off
        self.hist_feat = hist_feat              # Histogram features on or off
        self.hog_feat = hog_feat                # HOG features on or off
        self.y_start_stop = y_start_stop        # Min and max in y to search in slide_window()
        self.x_start_stop = x_start_stop
        self.xy_window = xy_window
        self.xy_overlap = xy_overlap
        self.heatmap_threshold = heatmap_threshold
        self.classifier = classifier

        self.stable_heatmaps = heatmap.StableHeatMaps(threshold=heatmap_threshold, num_frames=20)

    def detect(self, input_image):

        # Normalised copy of input image
        img = np.copy(input_image).astype(np.float32) / 255.

        slided_windows = window.slide_window(img, x_start_stop=self.x_start_stop,
                                             y_start_stop=self.y_start_stop,
                                             xy_window=self.xy_window, xy_overlap=self.xy_overlap)

        on_windows = window.search_windows(img, slided_windows, self.classifier, 
                                           color_space=self.color_space, spatial_size=self.spatial_size,
                                           hist_bins=self.hist_bins, orient=self.orient,
                                           pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                           hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                           hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        heat_map = self.stable_heatmaps.generate(img, on_windows)

        labels = label(heat_map)

        image_with_bb = window.draw_labeled_bboxes(input_image, labels)

        return image_with_bb