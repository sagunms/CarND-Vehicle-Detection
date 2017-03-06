import numpy as np
import matplotlib.image as mpimg

import argparse

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from vehicle_lib.config import *
from vehicle_lib import feature_extract
from vehicle_lib import utils
from vehicle_lib import debug

# Retrieve vehicle and non-vehicle training data
def prepare_train_features():

    vehicle_files_path = './data/vehicles/'
    non_vehicle_files_path = './data/non-vehicles/'

    # Load vehicle and non-vehicle images from fs to memory
    vehicle_files = utils.get_images_recursively(vehicle_files_path)
    vehicle_images = [mpimg.imread(file) for file in vehicle_files]

    non_vehicle_files = utils.get_images_recursively(non_vehicle_files_path)
    non_vehicle_images = [mpimg.imread(file) for file in non_vehicle_files]

    print('Number of vehicle files: {}'.format(len(vehicle_files)))
    print('Number of non-vehicle files: {}'.format(len(non_vehicle_files)))

    # Extract features for vehicle and non-vehicle images
    vehicle_features = feature_extract.extract_features(vehicle_images, 
                                                        config['color_space'], 
                                                        config['spatial_size'], 
                                                        config['hist_bins'], 
                                                        config['orient'], 
                                                        config['pix_per_cell'], 
                                                        config['cell_per_block'], 
                                                        config['hog_channel'], 
                                                        config['spatial_feat'], 
                                                        config['hist_feat'], 
                                                        config['hog_feat'])
    print('Shape of the vehicle features: {}'.format(vehicle_features.shape))

    non_vehicle_features = feature_extract.extract_features(non_vehicle_images, 
                                                            config['color_space'], 
                                                            config['spatial_size'], 
                                                            config['hist_bins'], 
                                                            config['orient'], 
                                                            config['pix_per_cell'], 
                                                            config['cell_per_block'], 
                                                            config['hog_channel'], 
                                                            config['spatial_feat'], 
                                                            config['hist_feat'], 
                                                            config['hog_feat'])
    print('Shape of the non-vehicle features: {}'.format(non_vehicle_features.shape))

    X_features = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
    print('Shape of the entire dataset: {}'.format(vehicle_features.shape))

    y_features = np.hstack((np.ones(len(vehicle_images)), np.zeros(len(non_vehicle_images))))

    return X_features, y_features

if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Vehicle Detection')
    parser.add_argument('-m', '--model', help='Model to save trained classifier to')
    args = parser.parse_args()

    # Prepare vehicle and non-vehicle features from training images
    X_features, y_features = prepare_train_features()

    # Split images into training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_features,
                                                        test_size=0.25, random_state=1024)

     # Initalise pipeline of Standard Scaler and Linear SVM classifier
    pipeline = Pipeline([('scaling', StandardScaler(with_mean=0, with_std=1)),
                         ('classifier', SVC(kernel='linear'))])

    # Train classifier
    pipeline.fit(X_train, y_train)

    # Measure validation accuracy
    accuracy = pipeline.score(X_test, y_test)
    print('Validation accuracy: {:.4f}'.format(accuracy))

    # Save model along with record of its config params
    joblib.dump({'model': pipeline, 'config': config}, args.model)

    print('Model saved: ', args.model)