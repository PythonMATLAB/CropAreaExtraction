"""canny算子提取边缘的探索、改进、对比。"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from math import ceil
from skimage.segmentation import mark_boundaries

if __name__ != "__main__":
    from .area import pixel_area


def show(list, titles=None, toptitle="show image"):
    """同时显示n个图像，自动确定多图的行列数。
    list = [image1, image2 ... , imagen]"""
    # 自动确定参数值
    if type(list) == np.ndarray:  # list是单个image，需要变成可迭代的list类型。
        list = [list]
    n = len(list)
    if not titles:
        titles = [""] * n
    figsize = (20, 10)
    if n <= 2:
        nrow = 1
        figsize = (15, 10)
    elif n <= 6:
        nrow = 2
    elif n <= 12:
        nrow = 3
    elif n <= 24:
        nrow = 4
    else:
        nrow = 5
    ncol = ceil(n / nrow)
    # 作图
    fig = plt.figure(toptitle, figsize=figsize)
    axes = fig.subplots(nrow, ncol, sharex=True, sharey=True)
    if n == 1:
        axes = np.array([axes])
    for ax, im, title in zip(axes.flat, list, titles):
        if len(im.shape) == 2:
            ax.imshow(im, cmap=plt.cm.gray)
        elif len(im.shape) == 3:
            ax.imshow(im)
        ax.axis("off")
        ax.set_title(title)
    fig.tight_layout()
    plt.show()


def show_canny(im, canny, title="show canny"):
    """显示canny算子提取的边缘"""
    fig = plt.figure(title, figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    if len(im.shape) == 2:
        ax.imshow(im, cmap=plt.cm.gray)
    elif len(im.shape) == 3:
        ax.imshow(im)
    ax.contour(canny, colors="red", linewidths=1)
    # ax.set_title('canny')
    plt.axis("off")
    plt.show()


def show_canny_contours(im, canny, title="show canny_contours"):
    """显示canny算子提取边缘的轮廓（等高线）"""
    fig = plt.figure(title, figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    if len(im.shape) == 2:
        ax.imshow(im, cmap=plt.cm.gray)
    elif len(im.shape) == 3:
        ax.imshow(im)
    canny_contours = measure.find_contours(canny.astype(int), 0.8, "high", "high")
    for contour in canny_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # ax.set_title('canny_contours')
    plt.axis("off")
    plt.show()


def show_segment(image, segment, title="show segment"):
    fig = plt.figure(title, figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segment))
    ax.set_title(title, size=12)
    plt.axis("off")
    plt.show()


# def show_3labels(image, labels, nrow=1, ncol=3):
#     # 作图
#     fig = plt.figure(figsize=(5 * ncol, 5 + 5 * nrow))
#     axes = fig.subplots(nrow, ncol, sharex=True, sharey=True)
#     if nrow == 1:
#         axes = np.array([axes])
#     for ax, im, label in zip(axes.flat, image, labels):
#         ax.imshow(mark_boundaries(im, label))
#         ax.axis("off")
#     axes[0][0].set_title("Easy")
#     axes[0][1].set_title("Medium")
#     axes[0][2].set_title("Hard")
#     fig.tight_layout()
#     plt.show()


def show_segments_area(image, segment, image_name, title="show segments area"):
    fig = plt.figure(title, figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segment))
    plt.axis("off")
    pixelarea = pixel_area(image_name)
    n = segment.max() - segment.min() + 1
    area = np.zeros((n, 3))
    for i in range(n):
        it = np.nditer(segment, flags=["multi_index"])
        while not it.finished:
            if segment[it.multi_index] == i:
                area[i][0] += it.multi_index[0]
                area[i][1] += it.multi_index[1]
                area[i][2] += 1
            it.iternext()
        area[i][0] /= area[i][2]
        area[i][1] /= area[i][2]
        area[i][2] *= pixelarea
    for i in range(n):
        plt.text(area[i, 0], area[i, 1], "{:.0f}".format(area[i, 2]), ha="center", va="bottom", fontsize=18, color="r")
    plt.show()


if __name__ == "__main__":
    from skimage.io import imread
    from skimage import color
    from skimage import feature
    from canny_connect import canny_connect
    from area import pixel_area

    image_name = "image_smallest.tif"
    im = imread(image_name)
    h = color.rgb2hsv(im)[:, :, 0]
    canny = feature.canny(h, sigma=1)
    canny_connection = canny_connect(canny.copy(), 9)
    i_canny = canny.copy()
    for i in np.nditer(i_canny, op_flags=["readwrite"]):
        i[...] = not i  # 取反

    show([h, canny, canny_connection, i_canny], ["h", "canny", "canny_connection", "i_canny"])
    show_canny(h, canny)
    show_canny_contours(im, canny)
    show_segment(im, segment=canny.astype(int))
    show_segments_area(im, canny.astype(int), image_name)
