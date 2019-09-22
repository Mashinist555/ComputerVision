import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import minimum_filter1d
from skimage.io import imread, imsave


def get_brightness(img):
    return np.add(
        np.multiply(img[:, :, 0], 0.299),
        np.add(
            np.multiply(img[:, :, 1], 0.587),
            np.multiply(img[:, :, 2], 0.114)
        )
    )


def get_energy(img):
    x_conv = np.array([[-1, 0, 1]])
    y_conv = np.array([[-1], [0], [1]])
    br_img = get_brightness(img)
    dx = convolve2d(br_img, x_conv, mode='same', boundary='symm')  # TODO boundary can be wrong
    dy = convolve2d(br_img, y_conv, mode='same', boundary='symm')  # TODO boundary can be wrong
    return np.sqrt(np.add(
        np.power(dx, 2),
        np.power(dy, 2)
    ))


def get_line_indices(img):
    energy = get_energy(img)
    imsave("test_energy.png", energy, )
    hei = energy.shape[0]
    wid = energy.shape[1]
    dp = np.zeros(energy.shape, dtype=np.float64)
    dp[0] = energy[0]
    for i in range(1, hei):
        dp[i] = np.add(energy[i], minimum_filter1d(dp[i - 1], 3))

    result = np.zeros(hei, dtype=int)
    pos = np.argmin(dp[hei - 1])
    result[hei - 1] = pos
    for i in range(hei - 2, -1, -1):
        pos = np.argmin(dp[i, max(0, pos - 1):min(pos + 2, wid)]) + max(0, pos - 1)
        result[i] = pos

    return result

def shrink(img):
    hei = img.shape[0]
    wid = img.shape[1]
    inds = get_line_indices(img)
    result = np.zeros((hei, wid-1, 3), dtype=np.float64)
    for i in range(0, hei):
        result[i] = np.concatenate((img[i,:inds[i]], img[i,inds[i]+1:]))
    imsave("test_result.png", result, )
    return result

def seam_carve(img, mode, mask):
    (orientation, direction) = mode.split(' ')
    if orientation == 'vertical':
        return shrink(np.transpose(img, (1,0,2)))
    else:
        return shrink(img)