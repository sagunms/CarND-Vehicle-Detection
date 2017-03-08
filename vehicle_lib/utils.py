import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def display_images(fname, n_images=12, n_rows=6, title=None):
    random_files = np.random.choice(fname, n_images)
    images = []
    for file in random_files:
        images.append(mpimg.imread(file))

    grid_space = gridspec.GridSpec(n_images // n_rows + 1, n_rows)
    grid_space.update(wspace=0.1, hspace=0.1)
    plt.figure(figsize=(n_rows, n_images // n_rows + 1))

    for i in range(0, n_images):
        ax1 = plt.subplot(grid_space[i])
        ax1.axis('off')
        ax1.imshow(images[i])

    if title is not None:
        plt.suptitle(title)
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