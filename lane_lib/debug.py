import numpy as np
import cv2
import matplotlib.pyplot as plt

# Debug helpers
# import pdb;pdb.set_trace()
def cv2_imshow(img, title=''):
    cv2.destroyAllWindows()
    cv2.imshow(title, img)

def plt_imshow(img, title=''):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off');
    plt.show()

def cv2_destroy():
    cv2.destroyAllWindows()