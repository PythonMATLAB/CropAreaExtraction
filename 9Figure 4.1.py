# =================================================
# @Author         : zly
# @Date           : 2024-04-27 09:15:12
# @LastEditTime   : 2024-04-28 19:44:23
# =================================================
from skimage.io import imread
from tool.feature import DrawFigure
from skimage.segmentation import mark_boundaries
import numpy as np


# 一、准备数据
easy = imread("0image_data/easy1200-0.1.tif")
medium = imread("0image_data/medium1000-0.1.tif")
hard = imread("0image_data/hard2000-0.1.tif")
easy_labels = imread("0image_data/easy1200-0.1_labels.tif")[:, :, 0]
medium_labels = imread("0image_data/medium1000-0.1_labels.tif")[:, :, 0]
hard_labels = imread("0image_data/hard2000-0.1_labels.tif")[:, :, 0]

images = [easy, medium, hard]
labels = [easy_labels, medium_labels, hard_labels]


# 二、作图
im = []
for i in range(len(images)):
    im += [mark_boundaries(images[i], labels[i], color=(1, 1, 0), outline_color=None)]

DrawFigure(im, ylabel=["R    "], nrows=1, wspace=0.01, title=["a.简单混合样方", "b.中等混合样方", "c.困难混合样方"])
