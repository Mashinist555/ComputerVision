from skimage.io import imread, imsave
import numpy as np

factor = 0.08

def cut(image):
    hei = image.shape[0]
    wid = image.shape[1]
    return image[int(hei * factor):int(hei * (1 - factor)), int(wid * factor):int(wid * (1 - factor))]


def mse(img1, img2):
    return np.sum(np.square(np.subtract(img1, img2))) / np.size(img1)


def imin(off):
    if off <= 0:
        return 0
    else:
        return off


def imax(max, off):
    if off >= 0:
        return max
    else:
        return max + off


def align(image, gcoords):
    hei = int(image.shape[0] / 3)
    wid = image.shape[1]
    offset = int(hei * (1+factor))
    r_img = cut(image[0:hei, :])
    g_img = cut(image[hei:hei * 2, :])
    b_img = cut(image[hei * 2:hei * 3, :])
    hei = r_img.shape[0]
    wid = r_img.shape[1]

    rcoord = (-100, -100)
    r_error = np.size(r_img) * 1234.0

    bcoord = (-100, -100)
    b_error = np.size(r_img) * 1234.0
    for dx in range(-15, 15):
        for dy in range(-15, 15):
            rerr = mse(g_img[imin(dx):imax(hei, dx), imin(dy):imax(wid, dy)],
                       r_img[imin(-dx):imax(hei, -dx), imin(-dy):imax(wid, -dy)])
            berr = mse(g_img[imin(dx):imax(hei, dx), imin(dy):imax(wid, dy)],
                       b_img[imin(-dx):imax(hei, -dx), imin(-dy):imax(wid, -dy)])
            if rerr < r_error:
                r_error = rerr
                rcoord = (dx, dy)
            if berr < b_error:
                b_error = berr
                bcoord = (dx, dy)
    reshei = hei - max(abs(rcoord[0] - bcoord[0]), max(rcoord[0], bcoord[0]))
    reswid = wid - max(abs(rcoord[1] - bcoord[1]), max(rcoord[1], bcoord[1]))

    maxx = max(0, rcoord[0], bcoord[0])
    maxy = max(0, rcoord[1], bcoord[1])
    result = np.dstack((
        b_img[imin(maxx - bcoord[0]):imin(maxx - bcoord[0]) + reshei,
        imin(maxy - bcoord[1]):imin(maxy - bcoord[1]) + reswid],

        g_img[imin(maxx - 0):imin(maxx - 0) + reshei, imin(maxy - 0):imin(maxy - 0) + reswid],

        r_img[imin(maxx - rcoord[0]):imin(maxx - rcoord[0]) + reshei,
        imin(maxy - rcoord[1]):imin(maxy - rcoord[1]) + reswid],
    ))
    rcoord = (-rcoord[0] + gcoords[0] + int(image.shape[0] * (0+factor) / 3) - offset, -rcoord[1] + gcoords[1])
    bcoord = (-bcoord[0] + gcoords[0] + int(image.shape[0] * (2+factor) / 3) - offset, -bcoord[1] + gcoords[1])
    # imsave("test_bilinear.png", result, )
    return result, rcoord, bcoord
