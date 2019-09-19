import numpy as np
import time
import math
from scipy.signal import convolve2d

current_milli_time = lambda: int(round(time.time() * 1000))


def get_bayer_masks(n_rows, n_cols):
    # res = np.tile([, [[1, 0], [0, 1]], [[0, 0], [1, 0]]], (n_rows, n_cols))
    rlayer = np.array([[0, 1], [0, 0]])
    glayer = np.array([[1, 0], [0, 1]])
    blayer = np.array([[0, 0], [1, 0]])
    res = np.dstack((rlayer, glayer, blayer))
    # print(res.shape)
    res = np.tile(res, (n_rows, n_cols, 1))
    # res = np.transpose(res, (1, 2, 0))
    res = res[:n_rows, :n_cols, :]
    # print(res.shape)
    return res
    # np.dstack()


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
    # print(colored_img.shape)
    colored_img = np.transpose(colored_img, (2, 0, 1))
    mask = np.transpose(mask, (2, 0, 1))
    res = np.zeros(colored_img.shape, dtype=int)
    for channel in range(0, 3):
        res[channel] = np.divide(convolve2d(colored_img[channel], filter, mode='same'), convolve2d(mask[channel], filter, mode='same'))
        res[channel][mask[channel] > 0] = colored_img[channel][mask[channel] > 0]

    res = np.transpose(res, (1, 2, 0))
    print(current_milli_time() - start)
    return res
    # for i in range(0, hei):
    #     for j in range(0, wid):
    #         for channel in range(0, 3):
    #             if mask[i, j, channel] > 0:
    #                 res[i, j, channel] = colored_img[i, j, channel]
    #             else:
    #                 slice = colored_img[max(0, i - 1):min(hei, i + 2), max(0, j - 1):min(wid, j + 2), channel]
    #                 mask_slice = mask[max(0, i - 1):min(hei, i + 2), max(0, j - 1):min(wid, j + 2), channel]
    #                 res[i, j, channel] = int(np.sum(slice) / np.count_nonzero(mask_slice))
    #
    # print(current_milli_time() - start)
    # return res


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

gcoords = np.empty((2, 2,), dtype=object)
rcoords = np.empty((2, 2,), dtype=object)
bcoords = np.empty((2, 2,), dtype=object)

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

for i in range(0, 2):
    for j in range(0, 2):
        gcoords[i][j] = np.argwhere(g[i][j] != 0)
        rcoords[i][j] = np.argwhere(r[i][j] != 0)
        bcoords[i][j] = np.argwhere(b[i][j] != 0)

operations = 0


def improved_interpolation(raw_img):
    hei = raw_img.shape[0]
    wid = raw_img.shape[1]
    start = current_milli_time()
    res = np.zeros((hei, wid, 3), dtype=int)
    np.co
    for i in range(0, int(hei / 2) + 1):
        # if i > 0 and i % 100 == 0:
        #     print(current_milli_time() - start)
        #     print(operations)
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
                        # res[x + dx, y + dy, 0] = calcsum(raw_img, x + dx, y + dy, r[dx][dy], rcoords[dx][dy])
                        # res[x + dx, y + dy, 1] = calcsum(raw_img, x + dx, y + dy, g[dx][dy], gcoords[dx][dy])
                        # res[x + dx, y + dy, 2] = calcsum(raw_img, x + dx, y + dy, b[dx][dy], bcoords[dx][dy])

            # if x < hei and y < wid:
            #     res[x, y, 0] = raw_img[x, y]
            #     res[x, y, 1] = calcsum(raw_img, x, y, g)
            #     res[x, y, 2] = calcsum(raw_img, x, y, b0)
            # if x < hei and y + 1 < wid:
            #     res[x, y + 1, 0] = calcsum(raw_img, x, y + 1, r1)
            #     res[x, y + 1, 1] = raw_img[x, y + 1]
            #     res[x, y + 1, 2] = calcsum(raw_img, x, y + 1, b1)
            # if x + 1 < hei and y < wid:
            #     res[x + 1, y, 0] = calcsum(raw_img, x + 1, y, r2)
            #     res[x + 1, y, 1] = raw_img[x + 1, y]
            #     res[x + 1, y, 2] = calcsum(raw_img, x + 1, y, b2)
            # if x + 1 < hei and y + 1 < wid:
            #     res[x + 1, y + 1, 0] = calcsum(raw_img, x + 1, y + 1, r3)
            #     res[x + 1, y + 1, 1] = calcsum(raw_img, x + 1, y + 1, g)
            #     res[x + 1, y + 1, 2] = raw_img[x + 1, y + 1]

    # print(current_milli_time() - start)
    return np.clip(res, 0, 255)


def calcsum(raw_img, x, y, filter, coords):
    global operations
    hei = raw_img.shape[0]
    wid = raw_img.shape[1]
    norm_fact = 0
    result = 0.0
    # print("{} {}".format(x, y))
    # for i in range(0, 5):
    #     for j in range(0, 5):
    mini = max(0, -(x - 2))
    minj = max(0, -(y - 2))
    maxi = min(5, hei - x + 2)
    maxj = min(5, wid - y + 2)
    slice = raw_img[x - 2 + mini:x - 2 + maxi, y - 2 + minj:y - 2 + maxj]
    factor = filter[mini:maxi, minj:maxj]
    res1 = int(np.sum(np.multiply(slice, factor)) / np.sum(factor))
    # for (i, j) in coords:
    #     operations = operations + 1
    #     if x + i - 2 >= 0 and x + i - 2 < hei and y + j - 2 >= 0 and y + j - 2 < wid:
    #         result += raw_img[x + i - 2, y + j - 2] * filter[i, j]
    #         norm_fact += filter[i, j]
    #         # print("{} {} filter:{}".format(i, j, filter[i, j]))
    # res2 = result / norm_fact
    return res1

def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype(float)
    img_gt = img_gt.astype(float)
    mse = np.sum(np.square(np.subtract(img_gt, img_pred))) / np.size(img_pred)
    if mse == 0:
        raise ValueError

    psnr = 10 * math.log10(math.pow(np.max(img_gt), 2) / mse)
    return psnr