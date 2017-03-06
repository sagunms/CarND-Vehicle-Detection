import os
import numpy as np
import argparse
from moviepy.editor import VideoFileClip

from sklearn.externals import joblib

from vehicle_lib import vehicle_detect
from vehicle_lib import debug

if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Advanced Lane Lines')
    # Example: python lanelines.py -i 'project_video.mp4' -o 'annotated_project_video.mp4'
    parser.add_argument('-i', '--inputfile', type=str, default='project_video.mp4',
                        help='Input driving video (recommended .mp4 file)')
    parser.add_argument('-o', '--outputfile', type=str, default='annotated_project_video.mp4',
                        help='Output video file')
    parser.add_argument('-m', '--model', help='model used to classify cars')
    args = parser.parse_args()

    if not os.path.exists(args.inputfile):
        raise ValueError('Input path not found.')

    # Load trained model from file
    data = joblib.load(args.model)
    pipeline = data['model']
    params = data['config']
    
    print('Model loaded: {}'.format(args.model))

    detector = vehicle_detect.VehicleDetect(color_space = params['color_space'],
											orient = params['orient'],
											pix_per_cell = params['pix_per_cell'],
											cell_per_block = params['cell_per_block'],
											hog_channel = params['hog_channel'],
											spatial_size = params['spatial_size'],
											hist_bins = params['hist_bins'],
											spatial_feat = params['spatial_feat'],
											hist_feat = params['hist_feat'],
											hog_feat = params['hog_feat'],
											y_start_stop = params['y_start_stop'],
											x_start_stop = params['x_start_stop'],
											xy_window = params['xy_window'],
											xy_overlap = params['xy_overlap'],
											heatmap_threshold = params['heat_threshold'],
											classifier=pipeline)

    # import pdb; pdb.set_trace()

    # Load video
    clip = VideoFileClip(args.inputfile)#.subclip(39, 41)
    # Detect vehicles from video
    out_clip = clip.fl_image(detector.detect)
    # Save annotated frames to output video file
    out_clip.write_videofile(args.outputfile, audio=False)