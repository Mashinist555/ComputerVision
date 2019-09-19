from skimage.io import imread, imsave
import numpy as np

def cut(image):
    factor = 0.08
    hei = image.shape[0]
    wid = image.shape[1]
    return image[int(hei * factor):int(hei*(1-factor)),int(wid * factor):int(wid*(1-factor))]

def mse(img1, img2):
    return np.sum(np.square(np.subtract(img1, img2)))/np.size(img1)

def align(image, gcoords):
    hei = int(image.shape[0] / 3)
    wid = image.shape[1]
    r_img = cut(image[0:hei, :])
    g_img = cut(image[hei:hei * 2, :])
    b_img = cut(image[hei * 2:hei * 3, :])
    hei = r_img.shape[0]
    wid = r_img.shape[1]

    rcoord = (-100, -100)
    r_error = np.size(r_img)*1234.0

    bcoord = (-100, -100)
    b_error = np.size(r_img) * 1234.0
    for dx in range(-15, 15):
        for dy in range(-15, 15):
            if dx<=0:
                if dy<=0:
                    rerr = mse(g_img[0:dx, 0:dy], r_img[dx:,dy:])
                    berr = mse(g_img[0:dx, 0:dy], b_img[dx:, dy:])
                else:
                    rerr = mse(g_img[0:dx, dy:], r_img[dx:, 0:-dy])
                    berr = mse(g_img[0:dx, dy:], b_img[dx:, 0:-dy])
            else:
                if dy<=0:
                    rerr = mse(g_img[dx, 0:dy], r_img[0:-dx:, dy:])
                    berr = mse(g_img[dx, 0:dy], b_img[0:-dx:, dy:])
                else:
                    rerr = mse(g_img[dx, dy], r_img[0:-dx:, 0:-dy])
                    berr = mse(g_img[dx, dy], b_img[0:-dx:, 0:-dy])
            if rerr < r_error:
                r_error = rerr
                rcoord = (dx, dy)
            if berr < b_error:
                b_error = berr
                bcoord = (dx, dy)
    # imsave("test_bilinear.png", g_img, )
    i=1
