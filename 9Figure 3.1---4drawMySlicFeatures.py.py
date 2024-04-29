# =================================================
# @Author         : zly
# @Date           : 2024-01-16 02:46:22
# @LastEditTime   : 2024-04-30 02:49:01
# =================================================
# 我的论文特征指标图v3
from skimage.io import imread
import numpy as np
from skimage import feature, color
from skimage import measure
from skimage.segmentation import mark_boundaries

from show.show import show, show_segment, show_canny
from tool.feature import get_feature
from tool.feature import canny_connect
from tool.feature import get_canny_mask
from tool.feature import get_canny_labels
from tool.feature import get_canny_labels2
from tool.feature import feature_names
from tool.feature import DrawAllFeatures

# 一、准备数据
easy = imread("0image_data/easy1200-0.1.tif")
medium = imread("0image_data/medium1000-0.1.tif")
hard = imread("0image_data/hard2000-0.1.tif")
images = [easy, medium, hard]

# 3个样方*36个特征指标
Original = [easy.astype(np.float32), medium.astype(np.float32), hard.astype(np.float32)]
features = get_feature(Original)
# print(len(features))


# 二、图3.1 ：36个特征指标和原始图像对比图
ylabel = ["R    ", "    G", "B    "]
ylabel += ["    H", "S    ", "    V"]
ylabel += ["RCC          ", "          GCC", "BCC          "]
ylabel += ["           GRRI", "RGRI           ", "           GBRI"]
ylabel += ["BGRI           ", "          RBRI", "BRRI           "]
ylabel += ["        ExB", "ExG         ", "            MExG"]
ylabel += ["ExR         ", "           ExGR", "GLI         "]
ylabel += ["              NGRDI", "MGRVI               ", "          VARI"]
ylabel += ["NRBDI               ", "              MRBVI", "NRGDI               "]
ylabel += ["             NGBDI", "RGBVI               ", "      WI"]
ylabel += ["TGI          ", "           CIVE", "VEG          "]
ylabel += ["                V-MSAVI", "COM1           ", "          COM2"]
DrawAllFeatures(images + images + features, ["Original               ", "               Original"] + ylabel)
# 图3.1-1 ：小图title在上面
DrawAllFeatures(
    images + images + features, ["Original               ", "               Original"] + ylabel, pos="lefttopright"
)


# 三、图3.1-2：36个特征指标在mycanny自适应sigma分割下的效果图
im = []
for i in range(len(features)):
    h = features[i]
    image = images[i % len(Original)]
    canny_labels = get_canny_labels2(h, sigma=0.8, remove_edges=False)
    im += [mark_boundaries(image, canny_labels)]
DrawAllFeatures(im, ylabel)

# 四、观察：12个特征指标在mycanny自适应sigma分割下的效果图 * 3
# ylabel = ["R    ", "    G", "B    "]
# ylabel += ["    H", "S    ", "    V"]
# ylabel += ["RCC          ", "          GCC", "BCC          "]
# ylabel += ["           GRRI", "RGRI           ", "           GBRI"]
# DrawAllFeatures(im[0:36], ylabel, wspace=0.01)
# ylabel = ["BGRI           ", "          RBRI", "BRRI           "]
# ylabel += ["        ExB", "ExG         ", "            MExG"]
# ylabel += ["ExR         ", "           ExGR", "GLI         "]
# ylabel += ["              NGRDI", "MGRVI               ", "          VARI"]
# DrawAllFeatures(im[36:72], ylabel, wspace=0.01)
# ylabel = ["NRBDI               ", "              MRBVI", "NRGDI               "]
# ylabel += ["             NGBDI", "RGBVI               ", "      WI"]
# ylabel += ["TGI          ", "           CIVE", "VEG          "]
# ylabel += ["                V-MSAVI", "COM1           ", "          COM2"]
# DrawAllFeatures(im[72:108], ylabel, wspace=0.01)
