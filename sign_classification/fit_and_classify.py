import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from scipy.signal import convolve2d
from sklearn.model_selection import train_test_split
from sklearn import svm
from skimage.transform import resize
import math
import time

EPS = 1e-8
CELL_NUM = 8
CELL_SIZE = 8

current_milli_time = lambda: int(round(time.time() * 1000))


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
    cell_hei = math.ceil(hei / CELL_NUM)
    cell_wid = math.ceil(wid / CELL_NUM)
    res_hei = math.ceil(hei / cell_hei)
    res_wid = math.ceil(wid / cell_wid)
    cells = np.zeros((res_hei, res_wid, 8))
    for i in range(res_hei):
        for j in range(res_wid):
            lens = gradien_len[i * cell_hei:(i + 1) * cell_hei, j * cell_wid:(j + 1) * cell_wid]
            dirs = gradien_dir[i * cell_hei:(i + 1) * cell_hei, j * cell_wid:(j + 1) * cell_wid]
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


def extract_data():
    table = pd.read_csv('public_tests/00_test_img_input/train_gt.csv')
    object_ids = table[['class_id', 'phys_id']].drop_duplicates()
    train_ids, test_ids = train_test_split(object_ids, test_size=0.2)  # TODO some objects may disappear
    train_id_set = set()
    for index, row in train_ids.iterrows():
        train_id_set.add((row.class_id, row.phys_id))
    # print(train_id_set)
    train_data = table.merge(train_ids, how='inner', left_on=['class_id', 'phys_id'], right_on=['class_id', 'phys_id'])
    test_data = table.merge(test_ids, how='inner', left_on=['class_id', 'phys_id'], right_on=['class_id', 'phys_id'])
    return train_data, test_data


def fit_and_classify(train_features, train_labels, test_features):
    model = svm.LinearSVC(max_iter=3000)
    model.fit(train_features, train_labels)
    return model.predict(test_features)


loaded = 0
total = current_milli_time()
on_blocks = 0
on_resize = 0


def extract_hog(image):
    """
    Exctracts HOG vector from the image

    :param image: (height, width, 3) np.ndarray: grb image
    :return: (N)  np.ndarray : HOG vector
    """
    global on_resize
    start = current_milli_time()
    resized = resize(image, (CELL_NUM * CELL_SIZE, CELL_NUM * CELL_SIZE))
    on_resize += current_milli_time() - start
    global loaded
    global on_blocks
    loaded += 1
    if loaded % 100 == 0:
        print("{} images converted".format(loaded))
        print("Total time: {}".format(current_milli_time() - total))
        print("On blocks: {}".format(on_blocks))
        print("On resize: {}".format(on_resize))

    start = current_milli_time()
    result = calc_blocks(resized).flatten()
    on_blocks += current_milli_time() - start
    return result
