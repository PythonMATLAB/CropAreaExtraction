# =================================================
# @Author         : zly
# @Date           : 2024-03-14 11:50:25
# @LastEditTime   : 2024-04-22 04:13:23
# =================================================
import math
import statistics
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def labels_remove_edges(labels_input, n=5):
    """labels_input里面的0值代表是edges，用5临近的众数替代。"""
    labels = labels_input.copy()
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] == 0:
                imin = i - n if i - n > 0 else 0
                imax = i + n if i + n < labels.shape[0] else labels.shape[0] - 1
                jmin = j - n if j - n > 0 else 0
                jmax = j + n if j + n < labels.shape[1] else labels.shape[1] - 1
                windows = labels[imin : imax + 1, jmin : jmax + 1]
                num_mode = statistics.mode(windows.reshape(-1).tolist())
                if num_mode == 0:
                    arr = [i for i in windows.reshape(-1).tolist() if i != 0]
                    num_mode = statistics.mode(arr)
                labels[i, j] = num_mode
    return labels


def chess_segment(image, n=7):
    chess_labels = image[:, :, 0].copy().astype(np.int32)

    i_chess = math.ceil(chess_labels.shape[0] / n)
    j_chess = math.ceil(chess_labels.shape[1] / n)

    for i in range(i_chess):
        for j in range(j_chess):
            chess_labels[i * n : i * n + n, j * n : j * n + n].fill(i * j_chess + j)
    return chess_labels


def segments_labels(image_labels, chess_segments):
    chess_labels = image_labels * 10 ** len(str(chess_segments.max())) + chess_segments

    segments = chess_labels.copy()
    unique_labels = np.unique(chess_labels)
    for i in range(len(unique_labels)):
        segments[chess_labels == unique_labels[i]] = i

    return segments


def compress_gray(img, levels=4):
    if img.dtype == "uint8":
        bins = np.linspace(0, 255, levels)
    else:
        img = img / img.max() * levels
        bins = np.linspace(img.min(), img.max(), levels)

    compress_gray = np.digitize(img, bins, right=True)
    gray = np.uint8(compress_gray)
    return gray


def piecewise(im_float, levels=4):
    """图像的像素分段映射到0-255"""
    im = im_float.astype(np.uint8)
    for i in range(levels):
        if np.percentile(im_float, i / levels * 100) < np.percentile(im_float, (i + 1) / levels * 100):
            label = np.where(
                (np.percentile(im_float, i / levels * 100) <= im_float)
                & (im_float < np.percentile(im_float, (i + 1) / levels * 100))
            )
        else:
            label = np.where(np.percentile(im_float, i / levels * 100) == im_float)
        im[label] = i

    im[im_float == np.percentile(im_float, 100)] = levels - 1
    return im


def graycomatrix_irregular(img_gray, segments, label, distances, angles, levels=None, symmetric=False, normed=False):
    img_gray = np.ascontiguousarray(img_gray)

    if levels is None:
        levels = 256

    P = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.uint32, order="C")

    for j in range(len(angles)):
        for i in range(len(distances)):
            d_x = round(distances[i] * angles[j][0])
            d_y = round(distances[i] * angles[j][1])
            P[:, :, i, j] = get_glcm_with_labels(img_gray, segments, label, d_x, d_y, levels)
    # 反向
    # make each GLMC symmetric
    if symmetric:
        Pt = np.transpose(P, (1, 0, 2, 3))
        P = P + Pt
    # normalize each GLCM
    if normed:
        P = P.astype(np.float64)
        glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums

    return P


def get_glcm_with_labels(img_gray, segments, label, d_x, d_y, levels):
    ret = np.zeros((levels, levels), dtype=np.uint32, order="C")
    (height, width) = img_gray.shape

    for i in range(height):
        for j in range(width):
            if segments[i][j] == label:
                rows = img_gray[i][j]
                # 正向
                if 0 <= i + d_y < height and 0 <= j + d_x < width:
                    if segments[i + d_y][j + d_x] == label:
                        cols_forward = img_gray[i + d_y][j + d_x]
                        ret[rows][cols_forward] += 1
                # 反向
                # make each GLMC symmetric
                # if symmetric:
                #     if 0 <= i - d_y < height and 0 <= j - d_x < width:
                #         if segments[i - d_y][j - d_x] == label:
                #             cols_reverse = img_gray[i - d_y][j - d_x]
                #             ret[rows][cols_reverse] += 1
    return ret


def texture(
    img_gray,
    segments,
    distances=[1, 2, 3, 4],
    angles=np.array([[1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 1], [-1 / math.sqrt(2), 1 / math.sqrt(2)]]),
    levels=None,
    symmetric=False,
    normed=False,
    piece=False,
):
    if levels is None:
        levels = 256
    else:
        img_gray = compress_gray(img_gray, levels)
        if piece:
            img_gray = piecewise(img_gray, levels)

    # zh = ["对比度", "相异性", "同质性", "二阶矩", "能量", "相关性",]
    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    features = np.zeros((segments.max() + 1, 2 + len(props), len(distances), len(angles)), dtype=np.float64)

    for label in range(segments.max() + 1):
        row_indices, col_indices = np.where(segments == label)
        segments_window = segments[row_indices.min() : row_indices.max() + 1, col_indices.min() : col_indices.max() + 1]
        img_gray_window = img_gray[row_indices.min() : row_indices.max() + 1, col_indices.min() : col_indices.max() + 1]
        P = graycomatrix_irregular(
            img_gray_window, segments_window, label, distances, angles, levels, symmetric, normed
        )

        for i in range(len(props)):
            features[label, i, :, :] = graycoprops(P, prop=props[i])

        features[label, i + 1, :, :] = np.mean(P, (0, 1))
        features[label, i + 2, :, :] = np.var(P, (0, 1), ddof=1)

    return features
