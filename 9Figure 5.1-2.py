# =================================================
# @Author         : zly
# @Date           : 2024-03-12 21:57:10
# @LastEditTime   : 2024-04-29 19:17:48
# =================================================
"""共生矩阵的步长、角度已经确定，不再用max"""
import cv2
import math
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import color
from skimage.io import imread
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# from note_classify.skimage_examples.auto_examples_python.segmentation.plot_perimeters import R # 做出一个很漂亮的图
from show.show import show, show_segment
from show.tool import chess_segment
from show.tool import segments_labels
from show.tool import texture
from tool.feature import get_feature
from tool.feature import get_canny_labels
from tool.feature import heatmap
from tool.feature import DrawFigure
from tool.super_pixels import super_pixels_slic
from tool.super_pixels import merge_econ

# 一、读取数据
easy = imread("0image_data/easy1200-0.1.tif")
medium = imread("0image_data/medium1000-0.1.tif")
hard = imread("0image_data/hard2000-0.1.tif")
easy_labels = imread("0image_data/easy1200-0.1_labels.tif")[:, :, 0]
medium_labels = imread("0image_data/medium1000-0.1_labels.tif")[:, :, 0]
hard_labels = imread("0image_data/hard2000-0.1_labels.tif")[:, :, 0]

easy_multi_labels = imread("0image_data/easy1200-0.1econ-50-0.4-0.5.tif")
medium_multi_labels = imread("0image_data/medium1000-0.1econ-50-0.4-0.5.tif")
hard_multi_labels = imread("0image_data/hard2000-0.1econ-50-0.4-0.5.tif")
hard_multi_labels = merge_econ(hard, hard_multi_labels, scale=30)

easy_superpixels_labels = super_pixels_slic(easy, scale=40, sigma=1, n_segments=400)
medium_superpixels_labels = super_pixels_slic(medium, scale=15, sigma=1, n_segments=400)
hard_superpixels_labels = super_pixels_slic(hard, scale=30, sigma=1, n_segments=400)

easy_canny_labels = get_canny_labels(color.rgb2hsv(easy)[:, :, 0])
medium_canny_labels = get_canny_labels(color.rgb2hsv(medium)[:, :, 0])
hard_canny_labels = get_canny_labels(color.rgb2hsv(hard)[:, :, 0])


# 二、分类
def classify(image, canny_labels, image_labels, classifier=RandomForestClassifier(), n=15, result="nofigure"):
    # 1、参数设置
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segments = segments_labels(canny_labels, chess_segment(image, n=n))

    sample_weight = np.unique(segments, return_counts=True)[1]
    distances = [1, 2, 3, 4]
    angles = np.array([[1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 1], [-1 / math.sqrt(2), 1 / math.sqrt(2)]])

    # 2、x值：计算特征值
    if "MultinomialNB" in str(type(classifier)):
        f = texture(img_gray, segments, [distances[3]], [angles[2]], levels=32)
    if "SVC" in str(type(classifier)):
        f = texture(img_gray, segments, [distances[0]], [angles[2]], levels=32)
    if "RandomForest" in str(type(classifier)):
        f = texture(img_gray, segments, [distances[0]], [angles[0]], levels=32)
    x = f[:, :, 0, 0]  # 输入特征
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    # 3、y值：处理分类后标签值
    # 读取每个sements标签的真实分类
    y_class = np.zeros((segments.max() + 1), np.uint8)
    for i in range(segments.max() + 1):
        y_class[i] = statistics.mode(image_labels[segments == i])
    y_indexes = np.array(range(len(y_class)))

    # 4、分类
    train_splits = train_test_split(x, y_indexes, test_size=0.3, random_state=0)
    x_train, x_test, y_train_indexes, y_test_indexes = train_splits
    y_train = y_class[y_train_indexes]
    y_test = y_class[y_test_indexes]
    y_test_weight = sample_weight[y_test_indexes]

    if "SVC" in str(type(classifier)):
        tick = [0.001, 0.01, 0.1, 1, 10, 100]
        params = {"C": tick, "gamma": tick}
        grid_search = GridSearchCV(SVC(), param_grid=params, cv=5)
        grid_search.fit(x_train, y_train)
        c = grid_search.best_params_["C"]
        gamma = grid_search.best_params_["gamma"]
        classifier = SVC(kernel="rbf", C=c, gamma=gamma)

    classifier.fit(x_train, y_train)
    y_test_pred = classifier.predict(x_test)
    flag = accuracy_score(y_test, y_test_pred, sample_weight=y_test_weight)

    # 5、分类结果图
    y_pred = classifier.predict(x)  # 所有预测类别标签值
    segments_pred = segments.copy()  # 每个像素点的预测：segments_pred
    for i in range(len(y_pred)):
        segments_pred[segments_pred == i] = y_pred[i]
    # show(segments_pred)
    # show_segment(image, segments_pred)

    # 6、返回结果
    if result == "figure":
        return segments_pred
    else:
        return flag


# 三、分类效果图
def get_class_labels():
    r11 = classify(easy, easy_canny_labels, easy_labels, classifier=MultinomialNB(), result="figure")
    r12 = classify(medium, medium_canny_labels, medium_labels, classifier=MultinomialNB(), result="figure")
    r13 = classify(hard, hard_canny_labels, hard_labels, classifier=MultinomialNB(), result="figure")
    r21 = classify(easy, easy_canny_labels, easy_labels, classifier=SVC(), result="figure")
    r22 = classify(medium, medium_canny_labels, medium_labels, classifier=SVC(), result="figure")
    r23 = classify(hard, hard_canny_labels, hard_labels, classifier=SVC(), result="figure")
    r31 = classify(easy, easy_canny_labels, easy_labels, classifier=RandomForestClassifier(), result="figure")
    r32 = classify(medium, medium_canny_labels, medium_labels, classifier=RandomForestClassifier(), result="figure")
    r33 = classify(hard, hard_canny_labels, hard_labels, classifier=RandomForestClassifier(), result="figure")
    labels9 = [r11, r12, r13, r21, r22, r23, r31, r32, r33]
    return labels9


# 测试效果
# r31 = classify(easy, easy_canny_labels, easy_labels, classifier=RandomForestClassifier())
# print(r31)
# r32 = classify(medium, medium_canny_labels, medium_labels, classifier=RandomForestClassifier())
# print(r32)
# r33 = classify(hard, hard_canny_labels, hard_labels, classifier=RandomForestClassifier())
# print(r33)

# 四、作出分类效果图
# 图5.1
DrawFigure(
    [easy, medium, hard] + [easy_labels, medium_labels, hard_labels],
    ["原始影像              ", "期望分类结果                    "],
    nrows=2,
    wspace=-0.3,
    title=["a.简单样方", "b.中等样方", "c.困难样方"],
    pos="leftbottom",
)

# 图5.2正式版
y1 = ["最大似然分类                     "]
y2 = ["支持向量分类                     "]
y3 = ["随机森林分类                     "]
ylabel = y1 + y2 + y3
DrawFigure(
    [easy, medium, hard] + get_class_labels(),
    ["原始影像              "] + ylabel,
    nrows=4,
    wspace=-0.7,
    title=["a.简单样方", "b.中等样方", "c.困难样方"],
    pos="leftbottom",
)

# 图5.2实验版
DrawFigure(
    get_class_labels(),
    ylabel,
    nrows=3,
    wspace=-0.5,
    title=["a.简单混合样方", "b.中等混合样方", "c.困难混合样方"],
    pos="leftbottom",
)
