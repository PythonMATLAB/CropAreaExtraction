# =================================================
# @Author         : zly
# @Date           : 2023-10-16 23:15:14
# @LastEditTime   : 2024-03-25 03:25:11
# =================================================
import numpy as np
from skimage import graph
from skimage.segmentation import slic


def merge_mean_color(graph, src, dst):
    graph.nodes[dst]["total color"] += graph.nodes[src]["total color"]
    graph.nodes[dst]["pixel count"] += graph.nodes[src]["pixel count"]
    graph.nodes[dst]["mean color"] = graph.nodes[dst]["total color"] / graph.nodes[dst]["pixel count"]


def weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]["mean color"] - graph.nodes[n]["mean color"]
    diff = np.linalg.norm(diff)
    return {"weight": diff}


def super_pixels_slic(image, scale=15, sigma=1, n_segments=400):
    # scale 分割后合并的阈值threshold，越大合并越多
    # sigma  高斯滤波的参数σ，即为正态分布的σ，越大越模糊
    # n_segments: 把图片分割成n个区域
    segments = slic(image, n_segments=n_segments, sigma=sigma, enforce_connectivity=True, start_label=1)
    g = graph.rag_mean_color(image, segments)
    labels = graph.merge_hierarchical(
        segments,
        g,
        thresh=scale,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_mean_color,
        weight_func=weight_mean_color,
    )
    return labels


def merge_econ(image, segments, scale=15):
    g = graph.rag_mean_color(image, segments)
    labels = graph.merge_hierarchical(
        segments,
        g,
        thresh=scale,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_mean_color,
        weight_func=weight_mean_color,
    )
    return labels
