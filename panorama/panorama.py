import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform, SimilarityTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv, svd
from math import sqrt
import random

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=1000):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """

    # your code here
    orb = ORB(n_keypoints=n_keypoints)
    orb.detect_and_extract(rgb2gray(img))
    return (orb.keypoints, orb.descriptors)


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    cx = np.mean(pointsh[0])
    cy = np.mean(pointsh[1])
    centered = np.stack((pointsh[0] - cx, pointsh[1] - cy))
    N = sqrt(2) / np.mean(np.sqrt(centered[0] * centered[0] + centered[1] * centered[1]))
    matrix = np.array([
        [N, 0, -N * cx],
        [0, N, -N * cy],
        [0, 0, 1]
    ])
    res_points = np.transpose(np.matmul(matrix, pointsh))[:, :2]
    return (matrix, res_points)


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    n = src_keypoints.shape[0]

    A = np.zeros((n * 2, 9))
    x = src[:, 0]
    y = src[:, 1]
    x1 = dest[:, 0]
    y1 = dest[:, 1]
    for i in range(n):
        A[i * 2] = np.array([-x[i], -y[i], -1, 0, 0, 0, x1[i] * x[i], x1[i] * y[i], x1[i]])
        A[i * 2 + 1] = np.array([0, 0, 0, -x[i], -y[i], -1, y1[i] * x[i], y1[i] * y[i], y1[i]])

    u, s, vt = svd(A)
    row = vt[8]
    row = row / np.mean(np.sqrt(row * row))
    H = np.reshape(row, (3, 3))
    result = np.matmul(np.matmul(inv(dest_matrix), H), src_matrix)
    return result


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors,
                     max_trials=1000, residual_threshold=1, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """

    # your code here
    matches = match_descriptors(src_descriptors, dest_descriptors)
    n = matches.shape[0]
    res_inds = [-1, -1, -1, -1]
    res_kol = 0
    for trial in range(max_trials):
        inds = random.sample(range(n), 4)
        h = find_homography(src_keypoints[matches[inds, 0]], dest_keypoints[matches[inds, 1]])
        projected = ProjectiveTransform(h)(src_keypoints[matches[:, 0]])
        dist = np.sqrt(
            np.power(projected[:, 0] - dest_keypoints[matches[:, 1], 0], 2) +
            np.power(projected[:, 1] - dest_keypoints[matches[:, 1], 1], 2))
        kol = np.sum(dist < residual_threshold)
        if kol > res_kol:
            res_kol = kol
            res_inds = inds

    h = find_homography(src_keypoints[matches[res_inds, 0]], dest_keypoints[matches[res_inds, 1]])
    transform = ProjectiveTransform(h)
    projected = transform(src_keypoints[matches[:, 0]])
    dist = np.sqrt(
        np.power(projected[:, 0] - dest_keypoints[matches[:, 1], 0], 2) +
        np.power(projected[:, 1] - dest_keypoints[matches[:, 1], 1], 2))
    inliers = matches[dist < residual_threshold]
    transform = ProjectiveTransform(find_homography(src_keypoints[inliers[:, 0]], dest_keypoints[inliers[:, 1]]))
    if return_matches:
        return transform, inliers
    else:
        return transform


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()
    cur_h = DEFAULT_TRANSFORM().params
    for i in range(center_index - 1, -1, -1):
        cur_h = np.matmul(forward_transforms[i].params, cur_h)
        result[i] = ProjectiveTransform(cur_h)

    cur_h = DEFAULT_TRANSFORM().params
    for i in range(center_index, image_count - 1):
        cur_h = np.matmul(cur_h, inv(forward_transforms[i].params))
        result[i + 1] = ProjectiveTransform(cur_h)

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
        """
    # your code here
    corners = tuple(get_corners(image_collection, simple_center_warps))
    bound = get_min_max_coords(corners)
    min_y = bound[0][0]
    min_x = bound[0][1]
    for warp in simple_center_warps:
        shift = SimilarityTransform(translation=(-min_x, -min_y))
        warp.params = np.matmul(shift.params, warp.params)
    shape = np.array([bound[1][1] - bound[0][1], bound[1][0] - bound[0][0]], dtype=int)
    return simple_center_warps, shape


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    # your code here
    warped_image = warp(image, rotate_transform_matrix(transform).inverse, output_shape=output_shape)
    mask = warp(np.ones(image.shape, dtype=np.bool8), rotate_transform_matrix(transform).inverse,
                output_shape=output_shape)
    return warped_image, np.asarray(mask, dtype=np.bool8)


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(np.append(output_shape, 3))
    result_mask = np.zeros(output_shape, dtype=np.bool8)

    # your code here
    n = len(image_collection)
    for i in reversed(range(n)):
        warped_image, mask = warp_image(image_collection[i], final_center_warps[i], output_shape)
        result[mask] = warped_image[mask]
    return result


def get_gaussian_pyramid(image, n_layers, sigma=1):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    # your code here
    result = [None] * n_layers
    img = image
    result[0] = img
    for i in range(1, n_layers):
        result[i] = gaussian(img, sigma)
        img = result[i]
    return result


def get_laplacian_pyramid(image, n_layers=5, sigma=1):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    # your code here
    gauss = get_gaussian_pyramid(image, n_layers, sigma)
    result = [None] * n_layers
    for i in range(n_layers - 1):
        result[i] = gauss[i] - gauss[i + 1]
    result[n_layers - 1] = gauss[n_layers - 1]
    return result


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def blend_images(img1, img2, mask, n_layers, image_sigma, merge_sigma):
    img1_pyr = get_laplacian_pyramid(img1, n_layers, image_sigma)
    img2_pyr = get_laplacian_pyramid(img2, n_layers, image_sigma)
    mask_pyr = get_gaussian_pyramid(mask, n_layers, merge_sigma)
    result = np.zeros(img1.shape)
    for i in range(n_layers):
        for channel in range(3):
            result[:,:,channel] += img1_pyr[i][:,:,channel] * np.subtract(1, mask_pyr[i])
            result[:,:,channel] += img2_pyr[i][:,:,channel] * mask_pyr[i]

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=5, image_sigma=1, merge_sigma=1):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    # your code here
    image_collection = increase_contrast(image_collection)
    warps_with_masks = [warp_image(img, warp, output_shape) for img, warp in zip(image_collection, final_center_warps)]
    corners = tuple(get_corners(image_collection, final_center_warps))
    result = np.zeros(np.append(output_shape, 3))
    result += warps_with_masks[0][0]
    n = len(image_collection)
    for i in range(1, n):
        max_left = int((corners[i-1][2, 0] + corners[i-1][3, 0])/2)
        min_right = int((corners[i][0, 0] + corners[i][1, 0])/2)
        mask = np.zeros(output_shape)
        mask[:,(max_left + min_right)//2:] = 1.0
        result = blend_images(result, warps_with_masks[i][0], mask, n_layers, image_sigma, merge_sigma)
    return result
