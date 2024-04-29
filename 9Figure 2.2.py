# =================================================
# @Author         : zly
# @Date           : 2024-04-27 09:15:12
# @LastEditTime   : 2024-04-28 10:09:44
# =================================================
from skimage.io import imread
from tool.feature import DrawFigure

# 一、准备数据
easy = imread("0image_data/easy1200-0.1.tif")
medium = imread("0image_data/medium1000-0.1.tif")
hard = imread("0image_data/hard2000-0.1.tif")

# 二、作图
Original = [easy, medium, hard]
DrawFigure(
    Original, ylabel=["R    "], nrows=1, wspace=0.01, title=["a.简单混合样方", "b.中等混合样方", "c.困难混合样方"]
)
