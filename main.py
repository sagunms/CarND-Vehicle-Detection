import os
import numpy as np
import cv2
import argparse
import pprint

from moviepy.editor import *

from sklearn.externals import joblib

from vehicle_lib import vehicle_detect
from vehicle_lib.debug import *
from vehicle_lib.config import *

from lane_lib import calib
from lane_lib import lane_detect
from lane_lib import masking
from lane_lib import perspective

if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Advanced Lane Lines')
    # Example: python lanelines.py -i 'project_video.mp4' -o 'annotated_project_video.mp4'
    parser.add_argument('-i', '--inputfile', type=str, default='project_video.mp4',
                        help='Input driving video (recommended .mp4 file)')
    parser.add_argument('-o', '--outputfile', type=str, default='annotated_project_video_lane.mp4',
                        help='Output video file')
    parser.add_argument('-m', '--model', help='model used to classify cars')
    args = parser.parse_args()

    if not os.path.exists(args.inputfile):
        raise ValueError('Input path not found.')

    # Load trained model from file
    data = joblib.load(args.model)
    pipeline = data['model']
    params = data['config']
    
    print('Model loaded: {}.'.format(args.model))

    # Load video
    video = VideoFileClip(args.inputfile)#.subclip(37.5, 37.8)

    # --- VEHICLE DETECTION ---
    
    vehicle_detector = vehicle_detect.VehicleDetect(color_space = params['color_space'],
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
                                                    heatmap_buffer_size = config['heat_buffer_size'], # TODO: Save this into the model.mdl
                                                    classifier=pipeline)

    print('Vehicle Detector loaded with following params:')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    # --- LANE DETECTION ---
    
    # Get params only used for lane lines (lane_lib.config - model.config)
    lane_params = {k:v for k,v in config.items() if k not in params}

    # Instrinsic camera calibration
    calibrate = calib.CameraCalibrate()

    # Perspective transformation
    perspective = perspective.PerspectiveTransform(video.size[::-1])

    # Initialise lane line detector
    lane_detector = lane_detect.LaneDetect(perspective.M, perspective.Minv)

    print('Lane Detector loaded with following params:')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(lane_params)

    # --- COMBINED DETECTION ---
    
    def pipeline(img):

        # Find vehicles, enable video overlay
        img_vehicle = vehicle_detector.detect(img, frame_overlay=False)

        # Correct for camera distortion
        img = calibrate.undistort(img)

        # Find lane lines, disable video overlay
        img_lane = lane_detector.detect(img, frame_overlay=True)

        # Combine annotated outputs from both vehicle and lane line detectors
        img_combined = cv2.addWeighted(img_vehicle, 1.0, img_lane, 1.0, 0)

        return img_combined

    # import pdb; pdb.set_trace()
    
    # Detect vehicles from video
    outvideo = video.fl_image(pipeline)
    
    # # Iterate though input video stream
    # clip = []
    # for img in video.iter_frames(progress_bar = True):
    #     img_out = pipeline(img)
    #     # cv2.imshow('Frame', img_out)
    #     # cv2.waitKey(1)
    #     clip += [img_out]
    #     
    # # Write clips to output video file
    # outvideo = ImageSequenceClip(clip, fps=30)
    # outvideo.write_videofile(args.outputfile)

    # Save annotated frames to output video file
    outvideo.write_videofile(args.outputfile, audio=False)
