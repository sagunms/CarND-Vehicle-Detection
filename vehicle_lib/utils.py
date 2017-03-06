import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def get_images_recursively(parent, extension='.png'):
    file_container = []
    for root, dirs, files in os.walk(parent):
        for file in files:
            if file.endswith(extension):
                file_container.append(os.path.join(root, file))
    return file_container

def display_random_images(image_fname, n_images=12, images_per_row=6, main_title=None):
    random_files = np.random.choice(image_fname, n_images)
    images = []
    for file in random_files:
        images.append(img.imread(file))

    grid_space = gridspec.GridSpec(n_images // images_per_row + 1, images_per_row)
    grid_space.update(wspace=0.1, hspace=0.1)
    plt.figure(figsize=(images_per_row, n_images // images_per_row + 1))

    for i in range(0, n_images):
        ax1 = plt.subplot(grid_space[i])
        ax1.axis('off')
        ax1.imshow(images[i])

    if main_title is not None:
        plt.suptitle(main_title)
    plt.show()

def display_features(hog_features, images, color_map=None, suptitle=None):
    n_images = len(images)
    space = gridspec.GridSpec(n_images, 2)
    space.update(wspace=0.1, hspace=0.1)
    plt.figure(figsize=(4, 2 * (n_images // 2 + 1)))

    for i in range(0, n_images*2):
        if i % 2 == 0:
            ax1 = plt.subplot(space[i])
            ax1.axis('off')
            ax1.imshow(images[i // 2], cmap=color_map)
        else:
            ax2 = plt.subplot(space[i])
            ax2.axis('off')
            ax2.imshow(hog_features[i // 2], cmap=color_map)

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.show()

if __name__ == '__main__':
    import vehicle_lib
    import cv2

    vehicle_files_dir = './data/vehicles/'
    non_vehicle_files_dir = './data/non-vehicles/'

    vehicle_files = extract_files(vehicle_files_dir)
    non_vehicle_files = extract_files(non_vehicle_files_dir)

    print('Number of vehicle files: {}'.format(len(vehicle_files)))
    print('Number of non-vehicle files: {}'.format(len(non_vehicle_files)))
    image = img.imread(vehicle_files[0])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = vehicle.get_hog_features(gray, orient,
                                                   pix_per_cell, cell_per_block,
                                                   vis=True, feature_vec=False)

    # Plot the examples
    a = []
    b = []
    a.append(hog_image)
    b.append(gray)
    display_features(a, b)
