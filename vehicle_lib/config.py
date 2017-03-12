config = {
    'color_space'       : 'YCrCb',          # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient'            : 9,                # HOG orientations
    'pix_per_cell'      : 8,                # HOG pixels per cell
    'cell_per_block'    : 2,                # HOG cells per block
    'hog_channel'       : 'ALL',            # Can be 0, 1, 2, or "ALL"
    'spatial_size'      : (32, 32),         # Spatial binning dimensions
    'hist_bins'         : 32,               # Number of histogram bins
    'spatial_feat'      : True,             # Spatial features on or off
    'hist_feat'         : True,             # Histogram features on or off
    'hog_feat'          : True,             # HOG features on or off
    'x_start_stop'      : [None, None],
    'y_start_stop'      : [400, 600],
    'xy_window'         : (96, 85),
    'xy_overlap'        : (0.75, 0.75),
    'heat_threshold'    : 15,
    'heat_buffer_size'  : 20,
    'colour_filter_y'   : (90, 255),        # Lane finding
    'colour_filter_v'   : (10, 110),
    'sobel_grad_xy'     : 'x',
    'sobel_grad_xy_th'  : (5, 230),
    'sobel_grad_mag'    : 3,
    'sobel_grad_mag_th' : (20, 150),
    'sobel_grad_dir'    : 3,
    'sobel_grad_dir_th' : (0.5, 1.0)
};