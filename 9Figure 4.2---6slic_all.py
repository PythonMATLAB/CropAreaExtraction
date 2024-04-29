# =================================================
# @Author         : zly
# @Date           : 2024-03-12 21:57:10
# @LastEditTime   : 2024-04-30 03:24:58
# =================================================
import numpy as np
from skimage import color
from skimage.segmentation import mark_boundaries
from skimage.io import imread

from show.show import show, show_segment
from tool.feature import get_canny_labels
from tool.feature import get_feature0
from tool.feature import get_feature
from tool.feature import DrawFigure
from tool.super_pixels import super_pixels_slic
from tool.super_pixels import merge_econ


# 一、读取下采样影像和目视解译labels
easy = imread("0image_data/easy1200-0.1.tif")
medium = imread("0image_data/medium1000-0.1.tif")
hard = imread("0image_data/hard2000-0.1.tif")

easy_labels = imread("0image_data/easy1200-0.1_labels.tif")[:, :, 0]
medium_labels = imread("0image_data/medium1000-0.1_labels.tif")[:, :, 0]
hard_labels = imread("0image_data/hard2000-0.1_labels.tif")[:, :, 0]


# 二、计算分割labels
easy_canny_labels = get_canny_labels(color.rgb2hsv(easy)[:, :, 0])
medium_canny_labels = get_canny_labels(color.rgb2hsv(medium)[:, :, 0])
hard_canny_labels = get_canny_labels(color.rgb2hsv(hard)[:, :, 0])

easy_superpixels_labels = super_pixels_slic(easy, scale=40, sigma=1, n_segments=400)
medium_superpixels_labels = super_pixels_slic(medium, scale=15, sigma=1, n_segments=400)
hard_superpixels_labels = super_pixels_slic(hard, scale=30, sigma=1, n_segments=400)

easy_multi_labels = imread("0image_data/easy1200-0.1econ-50-0.4-0.5.tif")
medium_multi_labels = imread("0image_data/medium1000-0.1econ-50-0.4-0.5.tif")
hard_multi_labels = imread("0image_data/hard2000-0.1econ-50-0.4-0.5.tif")
hard__labels = merge_econ(hard, hard_multi_labels, scale=30)


# 三、3种分割方法的效果对比图
im1 = [
    mark_boundaries(easy, easy_multi_labels, color=(1, 1, 0)),
    mark_boundaries(medium, medium_multi_labels, color=(1, 1, 0)),
    mark_boundaries(hard, hard_multi_labels, color=(1, 1, 0)),
]
im2 = [
    mark_boundaries(easy, easy_superpixels_labels, color=(1, 1, 0)),
    mark_boundaries(medium, medium_superpixels_labels, color=(1, 1, 0)),
    mark_boundaries(hard, hard_superpixels_labels),
]
im3 = [
    mark_boundaries(easy, easy_canny_labels, color=(1, 1, 0)),
    mark_boundaries(medium, medium_canny_labels, color=(1, 1, 0)),
    mark_boundaries(hard, hard_canny_labels, color=(1, 1, 0)),
]
im = im1 + im2 + im3
y1 = ["多尺度分割                 "]
y2 = ["超像素分割                 "]
y3 = ["本文分割              "]
ylabel = y1 + y2 + y3
# Table 4.2
DrawFigure(
    im,
    ylabel=ylabel,
    nrows=3,
    wspace=-0.5,
    title=["a.简单混合样方", "b.中等混合样方", "c.困难混合样方"],
    pos="leftbottom",
)


# 四、分别观察3种分割方法的效果
im0 = [
    mark_boundaries(easy, easy_labels),
    mark_boundaries(medium, medium_labels),
    mark_boundaries(hard, hard_labels),
]
y0 = ["目视解译               "]
DrawFigure(im0, ylabel=y0, nrows=1, wspace=0.05, pos="leftbottom")
DrawFigure(im1, ylabel=y1, nrows=1, wspace=0.05, pos="leftbottom")
DrawFigure(im2, ylabel=y2, nrows=1, wspace=0.05, pos="leftbottom")
DrawFigure(im3, ylabel=y3, nrows=1, wspace=0.05, pos="leftbottom")
