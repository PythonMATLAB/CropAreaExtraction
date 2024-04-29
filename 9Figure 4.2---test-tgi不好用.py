# =================================================
# @Author         : zly
# @Date           : 2024-03-12 21:57:10
# @LastEditTime   : 2024-04-30 03:23:33
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
easy_canny_labels0 = get_canny_labels(color.rgb2hsv(easy)[:, :, 0])
easy_canny_labels = get_canny_labels(get_feature([easy.astype(np.float32)], 31)[0], sigma=3)
show_segment(easy, easy_canny_labels0)
show_segment(easy, easy_canny_labels)

# medium_canny_labels0 = get_canny_labels(color.rgb2hsv(medium)[:, :, 0])
# medium_canny_labels = get_canny_labels(get_feature([medium.astype(np.float32)], 31)[0], sigma=3)
# show_segment(medium, medium_canny_labels0)
# show_segment(medium, medium_canny_labels)

# hard_canny_labels0 = get_canny_labels(color.rgb2hsv(hard)[:, :, 0])
# hard_canny_labels = get_canny_labels(get_feature([hard.astype(np.float32)], 31)[0], sigma=3)
# show_segment(hard, hard_canny_labels0)
# show_segment(hard, hard_canny_labels)
