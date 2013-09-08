from skimage.filter import hsobel
from skimage.filter import vsobel
import math
import numpy as np

# return a vector of features by applying the Sobel operator
# on the image in the following format:
#   [ sobel_mag, sobel_angle, grad_x, grad_y ]
#
# This function applies the horizontal and vertical Sobel operators and
# calculate the magnitude and the gradient of the responses.
def sobel_features(image):
    h = hsobel(image)
    v = vsobel(image)
    Gx = np.sum(h)
    Gy = np.sum(v)
    m = math.sqrt(Gx**2 + Gy**2)
    g = 0
    if(Gx == 0):
        g = 0 if Gy == 0 else math.pi
    else:
        g = math.atan2(Gy,Gx)
    return [m, g, Gx, Gy]
