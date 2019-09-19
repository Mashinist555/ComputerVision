import numpy as np
import time
import math
from scipy.signal import convolve2d

current_milli_time = lambda: int(round(time.time() * 1000))


def get_bayer_masks(n_rows, n_cols):
    rlayer = np.array([[0, 1], [0, 0]])
    glayer = np.array([[1, 0], [0, 1]])
    blayer = np.array([[0, 0], [1, 0]])
    res = np.dstack((rlayer, glayer, blayer))
    res = np.tile(res, (n_rows, n_cols, 1))
    res = res[:n_rows, :n_cols, :]
    return res


def get_colored_img(raw_img):
    rows = raw_img.shape[0]
    cols = raw_img.shape[1]
    mask = get_bayer_masks(rows, cols)
    return np.multiply(mask, np.tile(np.expand_dims(raw_img, 2), (1, 1, 3)))


input = np.ones((5, 5))
filter = np.ones((3, 3))
res = convolve2d(input, filter)


def bilinear_interpolation(colored_img):
    start = current_milli_time()
    hei = colored_img.shape[0]
    wid = colored_img.shape[1]
    mask = get_bayer_masks(hei, wid)
    colored_img = np.transpose(colored_img, (2, 0, 1))
    mask = np.transpose(mask, (2, 0, 1))
    res = np.zeros(colored_img.shape, dtype=int)
    for channel in range(0, 3):
        res[channel] = np.divide(convolve2d(colored_img[channel], filter, mode='same'),
                                 convolve2d(mask[channel], filter, mode='same'))
        res[channel][mask[channel] > 0] = colored_img[channel][mask[channel] > 0]

    res = np.transpose(res, (1, 2, 0))
    print(current_milli_time() - start)
    return res


identity = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
g = np.zeros((2, 2, 5, 5), dtype=int)
g[0][0] = identity

g[1][0] = np.array([[0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [-1, 2, 4, 2, -1],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0]])

g[0][1] = g[1][0]
g[1][1] = identity

r = np.zeros((2, 2, 5, 5), dtype=int)
r[0][0] = np.array([[0, 0, 1, 0, 0],
                    [0, -2, 0, -2, 0],
                    [-2, 8, 10, 8, -2],
                    [0, -2, 0, -2, 0],
                    [0, 0, 1, 0, 0]])
r[0][1] = identity
r[1][0] = np.array([[0, 0, -3, 0, 0],
                    [0, 4, 0, 4, 0],
                    [-3, 0, 12, 0, -3],
                    [0, 4, 0, 4, 0],
                    [0, 0, -3, 0, 0]])
r[1][1] = np.transpose(r[0][0])

b = np.zeros((2, 2, 5, 5), dtype=int)
b[0][0] = r[1][1]
b[0][1] = r[1][0]
b[1][0] = identity
b[1][1] = r[0][0]


def improved_interpolation(raw_img):
    hei = raw_img.shape[0]
    wid = raw_img.shape[1]
    res = np.zeros((hei, wid, 3), dtype=int)
    for i in range(0, int(hei / 2) + 1):
        for j in range(0, int(wid / 2) + 1):
            x = i * 2
            y = j * 2
            for dx in range(0, 2):
                for dy in range(0, 2):
                    if x + dx < hei and y + dy < wid:
                        x0 = x + dx
                        y0 = y + dy
                        mini = max(0, -(x0 - 2))
                        minj = max(0, -(y0 - 2))
                        maxi = min(5, hei - x0 + 2)
                        maxj = min(5, wid - y0 + 2)
                        slice = raw_img[x0 - 2 + mini:x0 - 2 + maxi, y0 - 2 + minj:y0 - 2 + maxj]
                        if dx == 0 and dy == 1:
                            res[x + dx, y + dy, 0] = raw_img[x + dx, y + dy]
                        else:
                            factor = r[dx][dy][mini:maxi, minj:maxj]
                            res[x + dx, y + dy, 0] = int(np.sum(np.multiply(slice, factor)) / np.sum(factor))
                        if dx + dy == 0 or dx + dy == 2:
                            res[x + dx, y + dy, 1] = raw_img[x + dx, y + dy]
                        else:
                            factor = g[dx][dy][mini:maxi, minj:maxj]
                            res[x + dx, y + dy, 1] = int(np.sum(np.multiply(slice, factor)) / np.sum(factor))
                        if dx == 1 and dy == 0:
                            res[x + dx, y + dy, 2] = raw_img[x + dx, y + dy]
                        else:
                            factor = b[dx][dy][mini:maxi, minj:maxj]
                            res[x + dx, y + dy, 2] = int(np.sum(np.multiply(slice, factor)) / np.sum(factor))
    return np.clip(res, 0, 255)


def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype(float)
    img_gt = img_gt.astype(float)
    mse = np.sum(np.square(np.subtract(img_gt, img_pred))) / np.size(img_pred)
    if mse == 0:
        raise ValueError

    psnr = 10 * math.log10(math.pow(np.max(img_gt), 2) / mse)
    return psnr
