import numpy as np
import pandas as pd
from skimage import color
from skimage import feature
from skimage import segmentation
from skimage.io import imread
from scipy.spatial.distance import directed_hausdorff


from show.show import show, show_segment, show_canny
from tool.feature import get_canny_labels
from tool.feature import get_canny_labels2
from tool.feature import get_canny_mask
from tool.feature import get_feature
from tool.feature import piecewise
from tool.feature import feature_names
from tool.feature import surface_overlap
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
images = [easy, medium, hard]
Original = [easy.astype(np.float32), medium.astype(np.float32), hard.astype(np.float32)]


# 二、计算36个特征指标
Original = [easy.astype(np.float32)]
features = get_feature(Original)


# 三、计算表面距离
def cal_distances(image_labels, slic_labels, flag="uv"):
    image_mask = segmentation.find_boundaries(image_labels, mode="inner")
    indices = np.where(image_mask == True)
    u = np.array(list(zip(indices[0], indices[1])))

    slic_mask = segmentation.find_boundaries(slic_labels, mode="inner")
    indices = np.where(slic_mask == True)
    v = np.array(list(zip(indices[0], indices[1])))

    if flag == "uv":
        return directed_hausdorff(u, v)[0]
    elif flag == "vu":
        return directed_hausdorff(v, u)[0]
    elif flag == "overlap":
        return surface_overlap(image_labels, slic_labels)[0]
    else:
        return directed_hausdorff(u, v), surface_overlap(image_labels, slic_labels)


def get_distances(features, easy_labels, flag="uv"):
    results = np.zeros(len(features))
    ratio_segment = np.zeros(len(features))
    for i in range(len(features)):
        easy_canny_labels = get_canny_labels2(h=features[i])

        ratio_segment[i] = easy_canny_labels.max() / (easy_labels.max() + 1)
        if ratio_segment[i] < 1.0:
            ratio_segment[i] = 1 / ratio_segment[i]

        results[i] = cal_distances(easy_labels, easy_canny_labels, flag)
    return results, ratio_segment


# 四、计算三个样方*36个特征指标*2个评价距离的表面距离表
# easy
Original = [easy.astype(np.float32)]
features = get_feature(Original)
easy_uv, easy_ratio_segment = get_distances(features, easy_labels, flag="uv")
easy_vu = get_distances(features, easy_labels, flag="vu")[0]
easy_overlap = get_distances(features, easy_labels, flag="overlap")[0]
easy_uv_ranking = [easy_uv.argsort()[::1].tolist().index(i) + 1 for i in range(len(easy_uv))]
easy_vu_ranking = [easy_vu.argsort()[::1].tolist().index(i) + 1 for i in range(len(easy_vu))]
easy_overlap_ranking = [easy_overlap.argsort()[::-1].tolist().index(i) + 1 for i in range(len(easy_overlap))]
easy_ratio_ranking = [easy_ratio_segment.argsort()[::1].tolist().index(i) + 1 for i in range(len(easy_ratio_segment))]

# medium
Original = [medium.astype(np.float32)]
features = get_feature(Original)
medium_uv, medium_ratio_segment = get_distances(features, medium_labels, flag="uv")
medium_vu = get_distances(features, medium_labels, flag="vu")[0]
medium_overlap = get_distances(features, medium_labels, flag="overlap")[0]
medium_uv_ranking = [medium_uv.argsort()[::1].tolist().index(i) + 1 for i in range(len(medium_uv))]
medium_vu_ranking = [medium_vu.argsort()[::1].tolist().index(i) + 1 for i in range(len(medium_vu))]
medium_overlap_ranking = [medium_overlap.argsort()[::-1].tolist().index(i) + 1 for i in range(len(medium_overlap))]
medium_ratio_ranking = [
    medium_ratio_segment.argsort()[::1].tolist().index(i) + 1 for i in range(len(medium_ratio_segment))
]

# hard
Original = [hard.astype(np.float32)]
features = get_feature(Original)
hard_uv, hard_ratio_segment = get_distances(features, hard_labels, flag="uv")
hard_vu = get_distances(features, hard_labels, flag="vu")[0]
hard_overlap = get_distances(features, hard_labels, flag="overlap")[0]
hard_uv_ranking = [hard_uv.argsort()[::1].tolist().index(i) + 1 for i in range(len(hard_uv))]
hard_vu_ranking = [hard_vu.argsort()[::1].tolist().index(i) + 1 for i in range(len(hard_vu))]
hard_overlap_ranking = [hard_overlap.argsort()[::-1].tolist().index(i) + 1 for i in range(len(hard_overlap))]
hard_ratio_ranking = [hard_ratio_segment.argsort()[::1].tolist().index(i) + 1 for i in range(len(hard_ratio_segment))]


# 五、导出到Excel文件
# 导出到Excel文件：评价指标
data = {
    "特征指标": feature_names[0 : len(easy_uv)],
    "a-h(A,B)": easy_uv,
    "a-h(B,A)": easy_vu,
    "a-overlap": easy_overlap,
    "a-r": easy_ratio_segment,
    "b-h(A,B)": medium_uv,
    "b-h(B,A)": medium_vu,
    "b-overlap": medium_overlap,
    "b-r": medium_ratio_segment,
    "c-h(A,B)": hard_uv,
    "c-h(B,A)": hard_vu,
    "c-overlap": hard_overlap,
    "c-r": hard_ratio_segment,
}
df = pd.DataFrame(data)
df.to_excel("9Table2.1三个样方的3个距离2.xlsx", index=False)

# 导出到Excel文件：指标排序
data = {
    "特征指标": feature_names[0 : len(easy_uv)],
    "a-h(A,B)": easy_uv_ranking,
    "a-h(B,A)": easy_vu_ranking,
    "a-overlap": easy_overlap_ranking,
    "a-r": easy_ratio_ranking,
    "b-h(A,B)": medium_uv_ranking,
    "b-h(B,A)": medium_vu_ranking,
    "b-overlap": medium_overlap_ranking,
    "b-r": medium_ratio_ranking,
    "c-h(A,B)": hard_uv_ranking,
    "c-h(B,A)": hard_vu_ranking,
    "c-overlap": hard_overlap_ranking,
    "c-r": hard_ratio_ranking,
}
df = pd.DataFrame(data)
df.to_excel("9Table2.1参考-三个样方的3个距离的排序.xlsx", index=False)


# 导出到Excel文件：三排序优选指标
rankings = np.array(easy_uv_ranking) + np.array(easy_vu_ranking) + np.array(easy_overlap_ranking)
rankings += np.array(medium_uv_ranking) + np.array(medium_vu_ranking) + np.array(medium_overlap_ranking)
rankings += np.array(hard_uv_ranking) + np.array(hard_vu_ranking) + np.array(hard_overlap_ranking)
ranking_36_3_3 = [rankings.argsort()[::1].tolist().index(i) + 1 for i in range(len(rankings))]

ranking_36_3_3_ratio = (
    np.array(ranking_36_3_3)
    + np.array(easy_ratio_ranking)
    + np.array(medium_ratio_ranking)
    + np.array(hard_ratio_ranking)
)
ranking_all = [ranking_36_3_3_ratio.argsort()[::1].tolist().index(i) + 1 for i in range(len(ranking_36_3_3_ratio))]
data = {
    "特征指标": feature_names[0 : len(easy_uv)],
    "优选排名": ranking_all,
}
df = pd.DataFrame(data)
df.to_excel("9Table2.2优选排名2.xlsx", index=False)
