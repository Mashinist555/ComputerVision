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


def smart_align(base, img, scaling):
    hei = base.shape[0]
    wid = base.shape[1]
    mindx = -10
    maxdx = 10
    mindy = -10
    maxdy = 10
    if hei >= 200:
        newhei = hei // 2
        newwid = wid // 2
        newbase = np.divide(np.add(np.add(base[0:newhei * 2:2, 0:newwid * 2:2], base[0:newhei * 2:2, 1:newwid * 2:2]),
                                   np.add(base[1:newhei * 2:2, 0:newwid * 2:2], base[1:newhei * 2:2, 1:newwid * 2:2])),
                            4)
        newimg = np.divide(np.add(np.add(img[0:newhei * 2:2, 0:newwid * 2:2], img[0:newhei * 2:2, 1:newwid * 2:2]),
                                  np.add(img[1:newhei * 2:2, 0:newwid * 2:2], img[1:newhei * 2:2, 1:newwid * 2:2])),
                           4)
        (bdx, bdy) = smart_align(newbase, newimg, scaling * 2)
        mindx = bdx * 2 - 2
        maxdx = bdx * 2 + 2
        mindy = bdy * 2 - 2
        maxdy = bdy * 2 + 2

    coord = (-100, -100)
    error = np.size(img) * 1234.0
    for dx in range(mindx, maxdx):
        for dy in range(mindy, maxdy):
            curerr = mse(base[imin(dx):imax(hei, dx), imin(dy):imax(wid, dy)],
                         img[imin(-dx):imax(hei, -dx), imin(-dy):imax(wid, -dy)])
            if curerr < error:
                error = curerr
                coord = (dx, dy)

    return coord


def align(image, gcoords):
    hei = int(image.shape[0] / 3)
    wid = image.shape[1]
    offset = int(hei * (1 + factor))
    r_img = cut(image[0:hei, :])
    g_img = cut(image[hei:hei * 2, :])
    b_img = cut(image[hei * 2:hei * 3, :])
    hei = r_img.shape[0]
    wid = r_img.shape[1]

    rcoord = smart_align(g_img, r_img, 1)
    bcoord = smart_align(g_img, b_img, 1)
    reshei = hei - max(abs(rcoord[0] - bcoord[0]), max(abs(rcoord[0]), abs(bcoord[0])))
    reswid = wid - max(abs(rcoord[1] - bcoord[1]), max(abs(rcoord[1]), abs(bcoord[1])))

    maxx = max(0, rcoord[0], bcoord[0])
    maxy = max(0, rcoord[1], bcoord[1])
    result = np.dstack((
        b_img[imin(maxx - bcoord[0]):imin(maxx - bcoord[0]) + reshei,
        imin(maxy - bcoord[1]):imin(maxy - bcoord[1]) + reswid],

        g_img[imin(maxx - 0):imin(maxx - 0) + reshei, imin(maxy - 0):imin(maxy - 0) + reswid],

        r_img[imin(maxx - rcoord[0]):imin(maxx - rcoord[0]) + reshei,
        imin(maxy - rcoord[1]):imin(maxy - rcoord[1]) + reswid],
    ))
    rcoord = (-rcoord[0] + gcoords[0] + int(image.shape[0] * (0 + factor) / 3) - offset, -rcoord[1] + gcoords[1])
    bcoord = (-bcoord[0] + gcoords[0] + int(image.shape[0] * (2 + factor) / 3) - offset, -bcoord[1] + gcoords[1])
    # imsave("test_bilinear.png", result, )
    return result, rcoord, bcoord
