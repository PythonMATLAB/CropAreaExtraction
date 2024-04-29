# =================================================
# @Author         : zly
# @Date           : 2024-01-12 10:35:25
# @LastEditTime   : 2024-03-30 03:04:34
# =================================================
"""对比hsv图层"""
from skimage.io import imread
from skimage import color
import matplotlib.pylab as plt


def hsv(image_name):
    """对比hsv图层, HSV (Hue, Saturation, Value)."""
    # read as a ndarray
    im = imread(image_name)
    # 生成各种hsv图像
    hsvs = [
        color.rgb2hsv(im),
        color.rgb2hsv(im),
        color.rgb2hsv(im),
        color.rgb2hsv(im),
        color.rgb2hsv(im),
        color.rgb2hsv(im),
        color.rgb2hsv(im)[:, :, 0],
        color.rgb2hsv(im)[:, :, 1],
        color.rgb2hsv(im)[:, :, 2],
    ]
    hsvs[1][:, :, 1] = 0.5
    hsvs[2][:, :, 1] = 1
    hsvs[3][:, :, 2] = 0.5
    hsvs[4][:, :, 2] = 1
    hsvs[5][:, :, 1:] = 1
    titles = ["origin", "s=0.5", "s=1", "v=0.5", "v=1", "s=1,v=1", "h", "s", "v"]
    # 作图
    # fig = plt.figure("show hsv", figsize=(20, 10))
    # axes = fig.subplots(3, 3, subplot_kw={"xticks": [], "yticks": []})
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 10), subplot_kw={"xticks": [], "yticks": []})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax, image, title in zip(axes.flat, hsvs, titles):
        if len(image.shape) > 2:
            rgb = color.hsv2rgb(image)
            ax.imshow(rgb, cmap="gray")
        else:
            ax.imshow(image, cmap="gray")
        ax.set_title(title, size=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_name = "image_smallest.tif"
    hsv(image_name)
