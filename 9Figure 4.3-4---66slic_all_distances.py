# =================================================
# @Author         : zly
# @Date           : 2024-03-12 21:57:10
# @LastEditTime   : 2024-04-29 20:11:48
# =================================================
import numpy as np
from skimage import color
from skimage import segmentation
from skimage.io import imread
from scipy.spatial.distance import directed_hausdorff


from show.show import show, show_segment
from tool.feature import get_canny_labels
from tool.feature import get_canny_mask
from tool.feature import surface_overlap
from tool.feature import heatmap0
from tool.feature import heatmap
from tool.super_pixels import super_pixels_slic
from tool.super_pixels import merge_econ


# 一、查看目下采样视解译labels
easy = imread("0image_data/easy1200-0.1.tif")
medium = imread("0image_data/medium1000-0.1.tif")
hard = imread("0image_data/hard2000-0.1.tif")

easy_labels = imread("0image_data/easy1200-0.1_labels.tif")[:, :, 0]
medium_labels = imread("0image_data/medium1000-0.1_labels.tif")[:, :, 0]
hard_labels = imread("0image_data/hard2000-0.1_labels.tif")[:, :, 0]


# 二、计算分割labels
easy_multi_labels = imread("0image_data/easy1200-0.1econ-50-0.4-0.5.tif")
medium_multi_labels = imread("0image_data/medium1000-0.1econ-50-0.4-0.5.tif")
hard_multi_labels = imread("0image_data/hard2000-0.1econ-50-0.4-0.5.tif")
hard_multi_labels = merge_econ(hard, hard_multi_labels, scale=30)
# show(hard_multi_labels)
# show_segment(hard, hard_multi_labels, "canny_mask")

easy_superpixels_labels = super_pixels_slic(easy, scale=40, sigma=1, n_segments=400)
medium_superpixels_labels = super_pixels_slic(medium, scale=15, sigma=1, n_segments=400)
hard_superpixels_labels = super_pixels_slic(hard, scale=30, sigma=1, n_segments=400)

easy_canny_labels = get_canny_labels(color.rgb2hsv(easy)[:, :, 0])
medium_canny_labels = get_canny_labels(color.rgb2hsv(medium)[:, :, 0])
hard_canny_labels = get_canny_labels(color.rgb2hsv(hard)[:, :, 0])


# 三、计算表面距离
def distances(image_labels, slic_labels, h=None):
    image_mask = segmentation.find_boundaries(image_labels, mode="inner")
    # show(image_mask, ["image_mask"])
    # show(image_labels, ["image_labels"])
    indices = np.where(image_mask == True)
    u = np.array(list(zip(indices[0], indices[1])))

    if h is not None:
        slic_mask = get_canny_mask(h)
    else:
        slic_mask = segmentation.find_boundaries(slic_labels, mode="inner")

    # show(slic_labels, ["slic_labels"])
    indices = np.where(slic_mask == True)
    v = np.array(list(zip(indices[0], indices[1])))

    return directed_hausdorff(u, v), surface_overlap(image_labels, slic_labels)


r11 = distances(easy_labels, easy_multi_labels)
r12 = distances(medium_labels, medium_multi_labels)
r13 = distances(hard_labels, hard_multi_labels)
r21 = distances(easy_labels, easy_superpixels_labels)
r22 = distances(medium_labels, medium_superpixels_labels)
r23 = distances(hard_labels, hard_superpixels_labels)
r31 = distances(easy_labels, easy_canny_labels, color.rgb2hsv(easy)[:, :, 0])
r32 = distances(medium_labels, medium_canny_labels, color.rgb2hsv(medium)[:, :, 0])
r33 = distances(hard_labels, hard_canny_labels, color.rgb2hsv(hard)[:, :, 0])

results = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
print(r11)
print(r12)
print(r13)
print(r21)
print(r22)
print(r23)
print(r31)
print(r32)
print(r33)


# 四、热力图
distances = np.zeros((3, 3))
overlaps = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        distances[i][j] = results[i][j][0][0]
        overlaps[i][j] = results[i][j][1][0]

# 图4.3
heatmap(overlaps, savename="overlaps.png")
# 图4.4
heatmap(distances, savename="distances.png")
