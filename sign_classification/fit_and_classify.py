import numpy as np
from skimage.color import rgb2gray
from scipy.signal import convolve2d
import math

EPS = 1e-8

def calc_gradient(img):
    ix = convolve2d(img, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
    iy = convolve2d(img, np.array([[-1], [0], [1]]), mode='same', boundary='symm')
    gradient_len = np.sqrt(ix * ix + iy * iy)
    gradient_dir = np.arctan2(iy, ix)
    gradient_dir[gradient_dir < 0] = gradient_dir[gradient_dir < 0] + math.pi  # TODO check
    return gradient_len, gradient_dir


def calc_cell_bins(image):
    gray = rgb2gray(image)
    gradien_len, gradien_dir = calc_gradient(gray)
    hei = image.shape[0]
    wid = image.shape[1]
    chei = math.ceil(hei / 8)
    cwid = math.ceil(wid / 8)
    cells = np.zeros((chei, cwid, 8))
    for i in range(chei):
        for j in range(cwid):
            lens = gradien_len[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            dirs = gradien_dir[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            cells[i, j], _ = np.histogram(dirs, 8, range=(0, math.pi), weights=lens)

    return cells


def calc_blocks(image):
    cells = calc_cell_bins(image)
    horizontal = np.concatenate(
        [
            cells[1:, :, :],
            cells[:-1, :, :]
        ], axis=2)

    blocks = np.concatenate(
        [
            horizontal[:, 1:, :],
            horizontal[:, :-1, :]
        ], axis=2)
    norms = np.linalg.norm(blocks, axis=2) + EPS
    blocks /= np.expand_dims(norms, axis=2)
    return blocks


def fit_and_classify():
    pass


def extract_hog(image):
    """
    Exctracts HOG vector from the image

    :param image: (height, width, 3) np.ndarray: grb image
    :return: (N)  np.ndarray : HOG vector
    """
    return calc_blocks(image).flatten()
