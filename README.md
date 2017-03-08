# Vehicle Detection Project

### Overview

This project detects and tracks vehicles using traditional computer vision and machine learning techniques. These include performing a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of vehicle/non-vehicle images and training a Linear Support Vector Machines (SVM) classifier model. The algorithm extracts features from the input video stream by applying a colour transform, performing HOG, Colour Histogram and Spatial binning. Then a sliding-window technique is used to scan the road for vehicles in the images by using the trained classifier to indicate that certain patches correspond to a vehicle or not. False positive are filtered out and vehicle tracking is stabalised by thresholding on heat maps over a number of frames of overlapping bounding boxes. 

[//]: # (Image References)
[heatmaps]: ./output_images/heatmaps.png
[hog_features_non_vehicles]: ./output_images/hog_features_non_vehicles.png
[hog_features_vehicles]: ./output_images/hog_features_vehicles.png
[noisy_classifier_detections]: ./output_images/noisy_classifier_detections.png
[non_vehicle_images]: ./output_images/non_vehicle_images.png
[output_bboxes]: ./output_images/output_bboxes.png
[overview_gif]: ./output_images/overview.gif
[sliding_window_roi]: ./output_images/sliding_window_roi.png
[vehicle_images]: ./output_images/vehicle_images.png
[test_video]: ./annotated_project_video_test.mp4
[final_video]: ./annotated_project_video.mp4

The following animation demonstrate how the final model performs on the given video stream.

![alt text][overview_gif]

### Project goals

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

## Run Instructions 

The project is written in python and utilises numpy, OpenCV, scikit learn and MoviePy.

Here are the steps required to generate the model from scratch and run the project for vehicle tracking. 

#### Clone my project
```bash
git clone https://github.com/sagunms/CarND-Vehicle-Detection.git
cd CarND-Vehicle-Detection
```

#### Activate conda environment
Follow instructions from [CarND-Term1-Starter-Kit page](https://github.com/udacity/CarND-Term1-Starter-Kit) to setup the conda environment from scratch.
```bash
source activate carnd-term1
```

#### Download training data of vehicles and non-vehicles
```bash
mkdir data
cd data
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
unzip vehicles.zip
unzip non-vehicles.zip
cd ..
```

#### Configure parameters
```bash
vim vehicle_lib/config.py
```

#### Train model from downloaded training data
```bash
python model.py -m model.mdl
```

#### Run vehicle detection project (output video)
```bash
python main.py -m model.mdl -i project_video.mp4 -o annotated_project_video.mp4
```

---

## Project structure

### Source Code
The code is divided up into several files which are imported by model.py and main.py.
* main.py - Takes input video file, input trained model and outputs annotated video without bounding boxes highlighting the detected vehicles.
* model.py - Reads vehicle and non-vehicle labelled images, extracts HOG features for both classes, splits features into training and validation datasets. Then, trains a pipeline consisting of StandardScaler and a Linear SVM classifier and saves the trained model as output.
* vehicle_lib/config.py - Consists of configuration parameters such as colour space, HOG parameters, spatial size, histogram bins, sliding window parameters, and region of interest, etc.
* vehicle_lib/vehicle_detect.py - Main class of the project which encapsulates sliding windows, feature generation, svm, remove duplicates and false positives, etc.
* vehicle_lib/feature_extract.py - Consists of functions related to feature extraction such as HOG features, spatial binning, colour histogram, etc.
* vehicle_lib/window.py - Consists of functions related to sliding window traversal and predicting which patch of the video frame contains a vehicle using the trained SVM model. 
* vehicle_lib/heatmap.py - Class for stablising detected heatmaps. Maintains history of heat maps over multiple frames and takes aggregate of all frames. 
* vehicle_lib/utils.py - Consists of utility functions to display images, features, traverse through subdirectories to load training images.
* vehicle_lib/debug.py - Some plotting functions the helped during debugging. 


### Miscellaneous Files
 * VehicleDetection.ipynb - Jupyter notebook for generating various stages of the project to assist this writeup. Images produced from this notebook can also be found at output_images/*.png
 * model.mdl - Trained model saved as the outcome of training the Linear SVM classifier from model.py. This file was then used in main.py to produce the annotated videos for demonstrating the working of my vehicle detection project. 
 * annotated_project_video.mp4 - The output of the vehicle detection project when processing against project_video.mp4 video stream.
 * annotated_project_video_test.mp4 - The output of the vehicle detection project when processing against test_video.mp4 video stream. 

---

## Histogram of Oriented Gradients (HOG) and other features

First step is to extract the features used to train the classifier and then to classify the video frames.

The code for this step is contained in `extract_features` function in `feature_extract.py`. This is invoked by `prepare_train_features` function in `model.py`, which is ultimately invoked when runing the `__main__` to train the model. 

I started by reading in all the `vehicle` and `non-vehicle` labelled images and calling `extract_features` function.  Here is an example of some of the `vehicle` and `non-vehicle` classes:

![alt text][vehicle_images]

![alt text][non_vehicle_images]

After trying out different color spaces and different parameters `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

First I tried different parameters however, the one provided in the course material was better and therefore settled with that. I used `YCrCb` color space and HOG parameters of `orientations=9`, `pix_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_features_vehicles]

![alt text][hog_features_non_vehicles]

## Classifier

After extracting features, we need to train a classifier to be able to differentiate between a portion of the frame as being a vehicle or non-vehicle. 

I used `sklearn.pipeline.Pipeline()` to encapsulate both StandardScalar and linear Support Vector Machine (SVM) into one, train it and save it as a model file using `sklearn.externals.joblib`. This help separate training and prediction into `model.py` and `main.py` files respectively.

The `main.py` loads the saved model (`model.mdl`) and passes the loaded classifier pipeline to `VehicleDetect` class. 

The for training the classifier is contained in `extract_features` function in `feature_extract.py`. This is invoked by `prepare_train_features` function in `model.py`,which prepares vehicle and non-vehicle features from the provided training images and split into training and testing dataset at the ratio of 75% and 25% respectively. Initially, I experimented with various values of C parameter. However, I later found out that the default linear SVM parameters initialised achieved validation accuracy of 0.9903 which was sufficient for detecting vehicles from the video stream.

The classifier pipleine parameters are as follows:

```
Pipeline(steps=[('scaling', StandardScaler(copy=True, with_mean=0, with_std=1)), ('classifier', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
```

## Sliding Window Search

To locate the cars in each frame, a sliding window approach was used over a region of interest. Initially, I started with different window sizes and overlaps hoping to get a finer resolution and higher detection accuracy. I tried different window sizes, region of interest and overlaps. Through hit and trail, I settled for a more simplistic single scale window instead of varying the sizes. The parameters I used are as follows which can be found in `vehicle_lib/config.py`.

* `xy_window = (96, 96)`
* `xy_overlap = (.75, .75)`
* `y_start_stop = [400, 600]`
* `x_start_stop = [None, None]`

The region of interest for sliding window search includes only the portion of the road,  spanning from left to right. 

![alt text][sliding_window_roi]

For this project, I searched on YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. This was mostly based on recommended values available in computer vision literature. The parameters can be found in `vehicle_lib/config.py`. 

1. Spatial Binning
    * `spatial_size = (32, 32)`
    * Function: `bin_spatial()` in `vehicle_lib/feature_extract.py`.
2. Color Histograms
    * `hist_bins = 32`
    * Function: `color_hist()` in `vehicle_lib/feature_extract.py`.
3. Histogram of Oriented Gradients (HOG)
    * `pix_per_cell = 8`
    * `cell_per_block = 2`
    * `orient = 9`
    * `color_space = YCrCb`
    * Function: `get_hog_features()` in `vehicle_lib/feature_extract.py`.
    
## Filtering False positives and Vehicle Tracking

I recorded the positions of positive detections in each frame of the video. Numerous patches in the images are predicted as being a vehicle and therefore contains noisy false positives. 

![alt text][noisy_classifier_detections]

The above figure illustrates the need to filter out overlapping bounding boxes by filtering. For this, from the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a video frame. 

![alt text][heatmaps]

The bounding boxes then overlaid on the area of the blobs detected.

![alt text][output_bboxes]

This worked great for images but when testing with video frames, the bounding boxes fluctuated at different patches in the image. In order to achieve stable tracking of vehicles that were already detected in temporal dimension, I created a `StableHeatMaps` class in `vehicle_lib/heatmap.py` which maintains a historical sum of heat pixels (of same size as the input frame) over 20 frames. The class includes private methods `_add_heat()` which adds heat for all pixels that fall within the patch of positive detection by the classifier, and `_apply_threshold()` to remove noise. The method `generate()` generates an aggregate sum of heatmap over history of 20 frames which thereby helps to stabalise the predicted bounding boxes. 

## Video Implementation

The working implementation can be summarised with the following animation.

![alt text][overview_gif]

My pipeline should perform reasonably well on the entire project video. 

Here's a link to [test video result](./annotated_project_video_test.mp4) and [final project video result](./annotated_project_video.mp4).

---

## Discussion

This project was really exciting to work on but it's a shame I had very little time to work on it. The implementation is far from perfect, but vehicle detection works quite well for the given project video. However, several things could be improved.

My algorithm pipeline would probably fail for objects different from vehicles it was trained with such as motorbikes, cyclists and pedestrians. Perhaps if the training images from the same camera is used, the classifier accuracy would be better. 

Although summing heat map over several historical frames seems to stabilise tracking of vehicles, a more robust approach would probably be the application of Kalman Filters for vehicle tracking. 

One of the drawbacks is that my detection pipeline is very slow (~4.5s per frame) and therefore cannot be used for real-time applications. Recent deep learning techniques like YOLO seem better suited in terms of detection accuracy and performance and therefore would probably be worth evaluating as an alternative to traditional computer vision and machine learning techniques such as used in this project. 

A simple method of improving processing speed would be to drop frames or scan frames at high frequency over high confidence heatmaps and lower at other reason. Kalman filter again, would be better in tracking with lower computation. 

Had there been sufficient time, I would integrate Advanced Lane Lines detection into this project. It would also be interesting to integrate an additional pipeline for using Traffic line classification to detect road signs. 
