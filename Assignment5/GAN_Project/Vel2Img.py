import scipy.misc
import numpy as np

def velocityFieldToPng(frameArray):
    """ Returns an array that can be saved as png with scipy.misc.toimage
    from a velocityField with shape [height, width, 2]."""
    outputframeArray = np.zeros((frameArray.shape[0], frameArray.shape[1], 3))
    for x in range(frameArray.shape[0]):
        for y in range(frameArray.shape[1]):
            # values above/below 1/-1 will be truncated by scipy
            frameArray[y][x] = (frameArray[y][x] * 0.5) + 0.5
            outputframeArray[y][x][0] = frameArray[y][x][0]
            outputframeArray[y][x][1] = frameArray[y][x][1]
    return outputframeArray