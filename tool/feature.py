# =================================================
# @Author         : zly
# @Date           : 2024-01-12 18:45:30
# @LastEditTime   : 2024-04-30 03:03:52
# =================================================
import statistics
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from skimage import feature
from skimage import color
from skimage import measure


def flinear(im):
    """图像的像素值线性映射到0-1,plt作图会自动线性拉伸像素值，这个函数没有起到作用。"""
    min_value = np.min(im)
    max_value = np.max(im)
    if min_value == max_value:
        return im / 255
    else:
        return (im - min_value) / (max_value - min_value)


# def flinear255(im):
#     """图像的像素值线性映射到0-1,plt作图会自动线性拉伸像素值，这个函数没有起到作用。"""
#     min_value = np.min(im)
#     max_value = np.max(im)
#     if min_value == max_value:
#         return 255
#     else:
#         return np.round(255 * (im - min_value) / (max_value - min_value), 0)
#         return np.round(255 * (im - min_value) / (max_value - min_value), 0).astype(np.uint8)


def data_preprocessing(im, ncols=3):
    for i in range(len(im)):
        if i >= ncols:
            # 去除nan多是0/0造成的，替换为1.0
            im[i] = np.where(np.isnan(im[i]), 1.0, im[i])

            # 去除np.inf值=x/0，防止图片变黑只有个别白色像素点。
            im[i] = np.where(np.isinf(im[i]), 999999999.0, im[i])
            im[i][np.where((np.percentile(im[i], 255 / 256 * 100) <= im[i]))] = np.percentile(im[i], 255 / 256 * 100)

            # im[i] = img_as_ubyte(flinear(im[i])) # 这个会舍弃精度，导致hard图像的WI指数是黑白点。


def piecewise(im_float, levels=256):
    """图像的像素分段映射到0-255"""
    im = im_float.astype(np.uint8)
    for i in range(levels):
        if np.percentile(im_float, i / levels * 100) < np.percentile(im_float, (i + 1) / levels * 100):
            label = np.where(
                (np.percentile(im_float, i / levels * 100) <= im_float)
                & (im_float < np.percentile(im_float, (i + 1) / levels * 100))
            )
        else:
            label = np.where(np.percentile(im_float, i / levels * 100) == im_float)
        im[label] = i

    im[im_float == np.percentile(im_float, 100)] = levels - 1
    return im


def DrawFigure(
    im,
    ylabel,
    nrows=4,
    wspace=-0.82,
    hspace=0.05,
    ncols=3,
    title=["a.简单", "b.中等", "c.困难"],
    pos="bottom",
    PiecewiseLinear=False,
):

    # plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    matplotlib.rc("font", family="Microsoft YaHei", weight="bold")  # 设置中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    # fig, axes = plt.subplots(nrows, ncols, figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    if nrows == 1:
        axes = np.array([axes])

    for i in range(nrows):
        for j in range(ncols):
            # 作图
            # axes[i, j].imshow(im[i * ncols + j], cmap="bone")
            # axes[i, j].imshow(im[i * ncols + j], cmap="gray")
            if i != 0 and PiecewiseLinear == True:
                axes[i, j].imshow(piecewise(im[i * ncols + j]), cmap="gray")
            else:
                axes[i, j].imshow(im[i * ncols + j], cmap="gray")
                # axes[i, j].imshow(im[i * ncols + j], cmap="gray", vmin=0, vmax=250)

            if "top" in pos:
                # 首行图加标题,
                if i == 0:
                    axes[i, j].set_title(title[j], family="Times New Roman", size=12)
            if "bottom" in pos:
                # 末行图加x轴标签
                if i == nrows - 1:
                    axes[i, j].set_xlabel(title[j], size=12)
            if "left" in pos:
                # 首列图加左边y轴标签
                if j == 0:
                    axes[i, j].set_ylabel(ylabel[i], size=12, rotation="horizontal")

            # 设置坐标轴
            axes[i, j].xaxis.set_major_locator(plt.NullLocator())
            axes[i, j].yaxis.set_major_locator(plt.NullLocator())
            # 设置图片的边框为不显示
            axes[i, j].spines["right"].set_color("none")
            axes[i, j].spines["top"].set_color("none")
            axes[i, j].spines["left"].set_color("none")
            axes[i, j].spines["bottom"].set_visible(False)
    plt.show()


def DrawAllFeatures(im, ylabel, wspace=-0.78, pos="leftbottomright", PiecewiseLinear=False):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    # matplotlib.rc("font", family="Microsoft YaHei", weight="bold")  # 设置中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    ncols = 6
    nrows = len(im) // ncols
    title = ["a.简单", "b.中等", "c.困难"] * (ncols // 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2, 8), subplot_kw={"xticks": [], "yticks": []})
    left = 0.00
    bottom = 0.01
    right = 0.98
    top = 0.97
    hspace = 0.02
    if pos == "bottom":
        bottom = 0.03
        top = 0.99

    fig.subplots_adjust(left, bottom, right, top, wspace, hspace)

    if nrows == 1:
        axes = np.array([axes])

    for i in range(nrows):
        for j in range(ncols):
            # 作图
            # axes[i, j].imshow(im[i * ncols + j], cmap="bone")
            # axes[i, j].imshow(im[i * ncols + j], cmap="gray")
            if i != 0 and PiecewiseLinear == True:
                axes[i, j].imshow(piecewise(im[i * ncols + j]), cmap="gray")
            else:
                axes[i, j].imshow(im[i * ncols + j], cmap="gray")
                # axes[i, j].imshow(im[i * ncols + j], cmap="gray", vmin=0, vmax=250)

            if "top" in pos:
                # 首行图加标题,
                if i == 0:
                    axes[i, j].set_title(title[j], size=8)
            if "bottom" in pos:
                # 末行图加x轴标签
                if i == nrows - 1:
                    axes[i, j].set_xlabel(title[j], size=8)
            if "left" in pos:
                # 首列图加左边y轴标签
                if j == 0:
                    axes[i, j].set_ylabel(ylabel[i * 2], family="Times New Roman", size=12, rotation="horizontal")
            if "right" in pos:
                # 末列图加右边y轴标签
                if j == ncols - 1:
                    axes[i, j].yaxis.set_label_position("right")
                    axes[i, j].yaxis.tick_right()
                    axes[i, j].set_ylabel(ylabel[i * 2 + 1], family="Times New Roman", size=12, rotation="horizontal")

            # 设置坐标轴
            axes[i, j].xaxis.set_major_locator(plt.NullLocator())
            axes[i, j].yaxis.set_major_locator(plt.NullLocator())
            # 设置图片的边框为不显示
            axes[i, j].spines["right"].set_color("none")
            axes[i, j].spines["top"].set_color("none")
            axes[i, j].spines["left"].set_color("none")
            axes[i, j].spines["bottom"].set_visible(False)
    plt.show()


def get_feature0(Original, PiecewiseLinear=False):
    R = [image[:, :, 0] for image in Original]
    G = [image[:, :, 1] for image in Original]
    B = [image[:, :, 2] for image in Original]

    H = [color.rgb2hsv(image)[:, :, 0] for image in Original]
    S = [color.rgb2hsv(image)[:, :, 1] for image in Original]
    V = [color.rgb2hsv(image)[:, :, 2] for image in Original]

    RCC = [im[:, :, 0] / (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) for im in Original]
    GCC = [im[:, :, 1] / (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) for im in Original]
    BCC = [im[:, :, 2] / (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) for im in Original]

    GRRI = [im[:, :, 1] / im[:, :, 0] for im in Original]
    RGRI = [im[:, :, 0] / im[:, :, 1] for im in Original]
    GBRI = [im[:, :, 1] / im[:, :, 2] for im in Original]

    BGRI = [im[:, :, 2] / im[:, :, 1] for im in Original]
    RBRI = [im[:, :, 0] / im[:, :, 2] for im in Original]
    BRRI = [im[:, :, 2] / im[:, :, 0] for im in Original]

    ExB = [(1.4 * im[:, :, 2] - im[:, :, 1]) / (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) for im in Original]
    ExG = [2 * im[:, :, 1] - im[:, :, 0] - im[:, :, 2] for im in Original]
    MExG = [1.262 * im[:, :, 1] - 0.884 * im[:, :, 0] - 0.311 * im[:, :, 2] for im in Original]

    ExR = [1.4 * im[:, :, 0] - im[:, :, 1] for im in Original]
    ExGR = [im[:, :, 1] - 2.4 * im[:, :, 0] - im[:, :, 2] for im in Original]
    GLI = [
        (2 * im[:, :, 1] - im[:, :, 0] - im[:, :, 2]) / (2 * im[:, :, 1] + im[:, :, 0] + im[:, :, 2]) for im in Original
    ]

    NGRDI = [(im[:, :, 1] - im[:, :, 0]) / (im[:, :, 1] + im[:, :, 0]) for im in Original]
    MGRVI = [
        (np.square(im[:, :, 1]) - np.square(im[:, :, 0])) / (np.square(im[:, :, 1]) + np.square(im[:, :, 0]))
        for im in Original
    ]
    VARI = [(im[:, :, 1] - im[:, :, 0]) / (im[:, :, 1] + im[:, :, 0] - im[:, :, 2]) for im in Original]
    # VARI[-1] = [piecewise(im) for im in VARI[-1]]  # VARI hard不分段拉伸是一片灰

    NRBDI = [(im[:, :, 0] - im[:, :, 2]) / (im[:, :, 0] + im[:, :, 2]) for im in Original]
    MRBVI = [
        (np.square(im[:, :, 0]) - np.square(im[:, :, 2])) / (np.square(im[:, :, 0]) + np.square(im[:, :, 2]))
        for im in Original
    ]
    NRGDI = [(im[:, :, 0] - im[:, :, 1]) / (im[:, :, 0] + im[:, :, 1]) for im in Original]

    NGBDI = [(im[:, :, 1] - im[:, :, 2]) / (im[:, :, 1] + im[:, :, 2]) for im in Original]
    RGBVI = [
        (np.square(im[:, :, 1]) - im[:, :, 2] * im[:, :, 0]) / (np.square(im[:, :, 0]) + im[:, :, 2] * im[:, :, 0])
        for im in Original
    ]
    WI = [(im[:, :, 1] - im[:, :, 2]) / (im[:, :, 0] - im[:, :, 1]) for im in Original]
    # WI = [piecewise(im) for im in WI] # WI 不分段拉伸是一片灰

    # TGI = [-0.5 * (190 * (R[i] - G[i]) - 120 * (R[i] - B[i])) for i in range(len(R))]
    # CIVE = [0.441 * R[i] - 0.811 * G[i] + 0.385 * B[i] + 18.78745 for i in range(len(R))]
    # VEG = [G[i] / (R[i] ** 0.667 * B[i] ** 0.333) for i in range(len(R))]

    # V_MSAVI = [(2 * G[i] + 1 - np.sqrt((2 * G[i] + 1) ** 2 - 8 * (2 * G[i] - R[i] - B[i]))) / 2 for i in range(len(R))]

    TGI = [-0.5 * (190 * (im[:, :, 0] - im[:, :, 1]) - 120 * (im[:, :, 0] - im[:, :, 2])) for im in Original]
    CIVE = [0.441 * im[:, :, 0] - 0.811 * im[:, :, 1] + 0.385 * im[:, :, 2] + 18.78745 for im in Original]
    VEG = [im[:, :, 1] / (im[:, :, 0] ** 0.667 * im[:, :, 2] ** 0.333) for im in Original]

    V_MSAVI = [
        (2 * im[:, :, 1] + 1 - np.sqrt((2 * im[:, :, 1] + 1) ** 2 - 8 * (2 * im[:, :, 1] - im[:, :, 0] - im[:, :, 2])))
        / 2
        for im in Original
    ]
    COM1 = [ExG[i] + CIVE[i] + ExGR[i] + VEG[i] for i in range(len(R))]
    COM2 = [0.36 * ExG[i] + 0.417 * CIVE[i] + 0.17 * VEG[i] for i in range(len(R))]

    im = R + G + B + H + S + V + RCC + GCC + BCC + GRRI + RGRI + GBRI + BGRI + RBRI + BRRI + ExB + ExG + MExG
    im += ExR + ExGR + GLI + NGRDI + MGRVI + VARI + NRBDI + MRBVI + NRGDI + NGBDI + RGBVI + WI
    im += TGI + CIVE + VEG + V_MSAVI + COM1 + COM2

    data_preprocessing(im)
    if PiecewiseLinear:
        im[3:] = [piecewise(i) for i in im[3:]]

    return im
    return np.array(im)
    return np.array(im, dtype=object)


def get_feature(Original, i=0, PiecewiseLinear=False):
    if 35 == i or i == 36:
        R = [image[:, :, 0] for image in Original]
        ExG = [2 * im[:, :, 1] - im[:, :, 0] - im[:, :, 2] for im in Original]
        ExGR = [im[:, :, 1] - 2.4 * im[:, :, 0] - im[:, :, 2] for im in Original]
        CIVE = [0.441 * im[:, :, 0] - 0.811 * im[:, :, 1] + 0.385 * im[:, :, 2] + 18.78745 for im in Original]
        VEG = [im[:, :, 1] / (im[:, :, 0] ** 0.667 * im[:, :, 2] ** 0.333) for im in Original]

    if i == 1:
        R = [image[:, :, 0] for image in Original]
        im = R
    elif i == 2:
        G = [image[:, :, 1] for image in Original]
        im = G
    elif i == 3:
        B = [image[:, :, 2] for image in Original]
        im = B

    elif i == 4:
        H = [color.rgb2hsv(image)[:, :, 0] for image in Original]
        im = H
    elif i == 5:
        S = [color.rgb2hsv(image)[:, :, 1] for image in Original]
        im = S
    elif i == 6:
        V = [color.rgb2hsv(image)[:, :, 2] for image in Original]
        im = V

    elif i == 7:
        RCC = [im[:, :, 0] / (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) for im in Original]
        im = RCC
    elif i == 8:
        GCC = [im[:, :, 1] / (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) for im in Original]
        im = GCC
    elif i == 9:
        BCC = [im[:, :, 2] / (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) for im in Original]
        im = BCC

    elif i == 10:
        GRRI = [im[:, :, 1] / im[:, :, 0] for im in Original]
        im = GRRI
    elif i == 11:
        RGRI = [im[:, :, 0] / im[:, :, 1] for im in Original]
        im = RGRI
    elif i == 12:
        GBRI = [im[:, :, 1] / im[:, :, 2] for im in Original]
        im = GBRI

    elif i == 13:
        BGRI = [im[:, :, 2] / im[:, :, 1] for im in Original]
        im = BGRI
    elif i == 14:
        RBRI = [im[:, :, 0] / im[:, :, 2] for im in Original]
        im = RBRI
    elif i == 15:
        BRRI = [im[:, :, 2] / im[:, :, 0] for im in Original]
        im = BRRI

    elif i == 16:
        ExB = [(1.4 * im[:, :, 2] - im[:, :, 1]) / (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) for im in Original]
        im = ExB
    elif i == 17:
        ExG = [2 * im[:, :, 1] - im[:, :, 0] - im[:, :, 2] for im in Original]
        im = ExG
    elif i == 18:
        MExG = [1.262 * im[:, :, 1] - 0.884 * im[:, :, 0] - 0.311 * im[:, :, 2] for im in Original]
        im = MExG

    elif i == 19:
        ExR = [1.4 * im[:, :, 0] - im[:, :, 1] for im in Original]
        im = ExR
    elif i == 20:
        ExGR = [im[:, :, 1] - 2.4 * im[:, :, 0] - im[:, :, 2] for im in Original]
        im = ExGR
    elif i == 21:
        GLI = [
            (2 * im[:, :, 1] - im[:, :, 0] - im[:, :, 2]) / (2 * im[:, :, 1] + im[:, :, 0] + im[:, :, 2])
            for im in Original
        ]
        im = GLI

    elif i == 22:
        NGRDI = [(im[:, :, 1] - im[:, :, 0]) / (im[:, :, 1] + im[:, :, 0]) for im in Original]
        im = NGRDI
    elif i == 23:
        MGRVI = [
            (np.square(im[:, :, 1]) - np.square(im[:, :, 0])) / (np.square(im[:, :, 1]) + np.square(im[:, :, 0]))
            for im in Original
        ]
        im = MGRVI
    elif i == 24:
        VARI = [(im[:, :, 1] - im[:, :, 0]) / (im[:, :, 1] + im[:, :, 0] - im[:, :, 2]) for im in Original]
        # VARI[-1] = [piecewise(im) for im in VARI[-1]]  # VARI hard不分段拉伸是一片灰
        im = VARI

    elif i == 25:
        NRBDI = [(im[:, :, 0] - im[:, :, 2]) / (im[:, :, 0] + im[:, :, 2]) for im in Original]
        im = NRBDI
    elif i == 26:
        MRBVI = [
            (np.square(im[:, :, 0]) - np.square(im[:, :, 2])) / (np.square(im[:, :, 0]) + np.square(im[:, :, 2]))
            for im in Original
        ]
        im = MRBVI
    elif i == 27:
        NRGDI = [(im[:, :, 0] - im[:, :, 1]) / (im[:, :, 0] + im[:, :, 1]) for im in Original]
        im = NRGDI

    elif i == 28:
        NGBDI = [(im[:, :, 1] - im[:, :, 2]) / (im[:, :, 1] + im[:, :, 2]) for im in Original]
        im = NGBDI
    elif i == 29:
        RGBVI = [
            (np.square(im[:, :, 1]) - im[:, :, 2] * im[:, :, 0]) / (np.square(im[:, :, 0]) + im[:, :, 2] * im[:, :, 0])
            for im in Original
        ]
        im = RGBVI
    elif i == 30:
        WI = [(im[:, :, 1] - im[:, :, 2]) / (im[:, :, 0] - im[:, :, 1]) for im in Original]
        # WI = [piecewise(im) for im in WI] # WI 不分段拉伸是一片灰
        im = WI

    elif i == 31:
        TGI = [-0.5 * (190 * (im[:, :, 0] - im[:, :, 1]) - 120 * (im[:, :, 0] - im[:, :, 2])) for im in Original]
        im = TGI
    elif i == 32:
        CIVE = [0.441 * im[:, :, 0] - 0.811 * im[:, :, 1] + 0.385 * im[:, :, 2] + 18.78745 for im in Original]
        im = CIVE
    elif i == 33:
        VEG = [im[:, :, 1] / (im[:, :, 0] ** 0.667 * im[:, :, 2] ** 0.333) for im in Original]
        im = VEG

    elif i == 34:
        V_MSAVI = [
            (
                2 * im[:, :, 1]
                + 1
                - np.sqrt((2 * im[:, :, 1] + 1) ** 2 - 8 * (2 * im[:, :, 1] - im[:, :, 0] - im[:, :, 2]))
            )
            / 2
            for im in Original
        ]
        im = V_MSAVI
    elif i == 35:
        COM1 = [ExG[i] + CIVE[i] + ExGR[i] + VEG[i] for i in range(len(R))]
        im = COM1
    elif i == 36:
        COM2 = [0.36 * ExG[i] + 0.417 * CIVE[i] + 0.17 * VEG[i] for i in range(len(R))]
        im = COM2
    elif i == 0:
        im = []
        for j in range(1, 37):
            if j <= 3:
                im += get_feature(Original, j, PiecewiseLinear=False)
            if j > 3:
                if PiecewiseLinear:
                    im += get_feature(Original, j, PiecewiseLinear=True)
                else:
                    im += get_feature(Original, j, PiecewiseLinear=False)
        return im

    data_preprocessing(im)
    if PiecewiseLinear:
        im = [piecewise(i) for i in im]
    return im


def get_best_feature(image, PiecewiseLinear=False):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    TGI = -0.5 * (190 * (R - G) - 120 * (R - B))
    # CIVE = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    # VEG = G / (R ** 0.667 * B ** 0.333)

    im = TGI
    data_preprocessing(im)
    if PiecewiseLinear:
        im = piecewise(im)
    return im


feature_names = np.array(
    [
        "R",
        "G",
        "B",
        "H",
        "S",
        "V",
        "RCC",
        "GCC",
        "BCC",
        "GRRI",
        "RGRI",
        "GBRI",
        "BGRI",
        "RBRI",
        "BRRI",
        "ExB",
        "ExG",
        "MExG",
        "ExR",
        "ExGR",
        "GLI",
        "NGRDI",
        "MGRVI",
        "VARI",
        "NRBDI",
        "MRBVI",
        "NRGDI",
        "NGBDI",
        "RGBVI",
        "WI",
        "TGI",
        "CIVE",
        "VEG",
        "V_MSAVI",
        "COM1",
        "COM2",
    ]
)


def labels_remove_edges(labels_input, n=5):
    """labels_input里面的0值代表是edges，用5临近的众数替代。"""
    labels = labels_input.copy()
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] == 0:
                imin = i - n if i - n > 0 else 0
                imax = i + n if i + n < labels.shape[0] else labels.shape[0] - 1
                jmin = j - n if j - n > 0 else 0
                jmax = j + n if j + n < labels.shape[1] else labels.shape[1] - 1
                windows = labels[imin : imax + 1, jmin : jmax + 1]
                num_mode = statistics.mode(windows.reshape(-1).tolist())
                if num_mode == 0:
                    arr = [i for i in windows.reshape(-1).tolist() if i != 0]
                    num_mode = statistics.mode(arr)
                labels[i, j] = num_mode
    return labels


def canny_connect(im_in, k):
    """canny算子提取边缘，部分边缘不连通。此函数把边缘连通起来。"""
    im = im_in.copy()
    k = k // 2
    m, n = np.mgrid[: im.shape[0], : im.shape[1]]
    for i, j in zip(m.flat, n.flat):  # 连接端点到图像边缘
        if _endpoint(im, i, j):
            xarray, yarray = np.mgrid[
                max([0, i - k]) : min([i + k, im.shape[0] - 1]) + 1, max([0, j - k]) : min([j + k, im.shape[1] - 1]) + 1
            ]
            nearest = (2 * k + 3) ** k
            xnearest = i
            ynearest = j
            for x, y in zip(xarray.flat, yarray.flat):  # k范围内的边缘
                confine = x in [0, im.shape[0] - 1] or y in [0, im.shape[1] - 1]
                if confine:
                    distance = ((x - i) ** 2 + (y - j) ** 2) ** 0.5
                    if distance < nearest:  # k范围内的最近边缘
                        nearest = distance
                        xnearest = x
                        ynearest = y
            x = xnearest
            y = ynearest
            if x != i and y == j:  #
                im[range(i, x + np.sign(x - i), np.sign(x - i)), y] = im[i, j]
            if y != j and x == i:
                im[x, range(j, y + np.sign(y - j), np.sign(y - j))] = im[i, j]
    image = im.copy()
    # labels = measure.label(im, connectivity=2)   旧方法
    for i, j in zip(m.flat, n.flat):  # 遍历整个图像，连接端点到最近的边界
        confine = i in [0, im.shape[0] - 1] or j in [0, im.shape[1] - 1]
        if _endpoint(im, i, j) and not confine:  # 寻找非边缘端点，以便后面进行连接
            # print(i, j)
            xmin = max([0, i - k])
            xmax = min([i + k, im.shape[0] - 1])
            ymin = max([0, j - k])
            ymax = min([j + k, im.shape[1] - 1])
            label = measure.label(im[xmin : xmax + 1, ymin : ymax + 1], connectivity=2)
            xarray, yarray = np.mgrid[xmin : xmax + 1, ymin : ymax + 1]
            nearest = (2 * k + 3) ** k
            xnearest = i
            ynearest = j
            for x, y in zip(xarray.flat, yarray.flat):  # k范围内的1范围外的边界
                condition1 = im[x, y] and (abs(x - i) > 1 or abs(y - j) > 1)
                condition2 = label[x - xmin, y - ymin] != label[i - xmin, j - ymin]
                if condition1 and condition2:
                    distance = ((x - i) ** 2 + (y - j) ** 2) ** 0.5
                    if distance < nearest:  # k范围内的最近其他边界
                        nearest = distance
                        xnearest = x
                        ynearest = y
            x = xnearest
            y = ynearest
            if x != i and y == j:
                image[range(i, x + np.sign(x - i), np.sign(x - i)), y] = im[i, j]
            if y != j and x == i:
                image[x, range(j, y + np.sign(y - j), np.sign(y - j))] = im[i, j]
            if x != i and y != j:
                if abs(x - i) == abs(y - j):
                    d = abs(x - i)
                    for t in range(1, d):
                        image[i + np.sign(x - i) * t, j + np.sign(y - j) * t] = im[i, j]
                elif abs(x - i) < abs(y - j):
                    d = abs(x - i)
                    for t in range(1, d + 1):
                        image[i + np.sign(x - i) * t, j + np.sign(y - j) * t] = im[i, j]
                    image[x, range(j + np.sign(y - j) * (d + 1), y, np.sign(y - j))] = im[i, j]
                else:
                    d = abs(y - j)
                    for t in range(1, d + 1):
                        image[i + np.sign(x - i) * t, j + np.sign(y - j) * t] = im[i, j]
                    image[range(i + np.sign(x - i) * (d + 1), x, np.sign(x - i)), y] = im[i, j]
    return image


def _endpoint(im, i, j):
    """判断是否是边缘结束点，被connect调用。"""
    xarray, yarray = np.mgrid[
        max([0, i - 1]) : min([i + 1, im.shape[0] - 1]) + 1, max([0, j - 1]) : min([j + 1, im.shape[1] - 1]) + 1
    ]
    points = []
    for x, y in zip(xarray.flat, yarray.flat):  # 搜索k范围内的
        if im[x, y] and (abs(x - i) > 0 or abs(y - j) > 0):  # k范围内除中心外的边界点
            points.append([x, y])
    if im[i, j]:
        if len(points) == 1:
            return True
        elif len(points) == 2:
            distance = ((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2) ** 0.5
            if distance == 1:
                return True
            else:
                return False
        elif len(points) == 3:
            if points[0][0] == points[1][0] == points[2][0] or points[0][1] == points[1][1] == points[2][1]:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def get_canny_mask(h, sigma=0.8):
    """canny边缘分割，获取canny_mask,True:边界，False:非边界"""
    canny_mask = canny_connect(feature.canny(h, sigma), 7)
    return canny_mask


def get_canny_labels(h, sigma=0.8, remove_edges=True):
    """canny边缘分割，获取canny_labels：1、2、3……各代表一个连通区域"""
    canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)
    if remove_edges:
        canny_labels = labels_remove_edges(canny_labels)
    return canny_labels


def get_canny_labels2(h, sigma=0.8, remove_edges=True):
    """canny边缘分割，获取canny_labels：1、2、3……各代表一个连通区域"""
    canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)

    # while canny_labels.max() < 10:
    #     sigma += 5
    #     canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)

    if canny_labels.max() > 200:
        h = flinear(h)
        sigma = 3.0
        canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)
        while canny_labels.max() < 4:
            sigma -= 1
            canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)
            if sigma <= 0:
                break
    # while canny_labels.max() > 148:
    #     sigma += 0.5
    #     canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)

    if canny_labels.max() < 4:
        sigma -= 0.8
        canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)

    if remove_edges:
        canny_labels = labels_remove_edges(canny_labels)
    return canny_labels


# def get_canny_labels3(h, sigma=0.8, remove_edges=True):
#     """canny边缘分割，获取canny_labels：1、2、3……各代表一个连通区域"""
#     canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)

#     # while canny_labels.max() < 10:
#     #     sigma += 5
#     #     canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)

#     if np.unique(canny_labels, return_counts=True)[1].max() < 500:
#         h = flinear(h)
#         sigma = 3.0
#         canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)
#         while np.unique(canny_labels, return_counts=True)[1].max() > 10000:
#             sigma -= 1
#             canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)
#             if sigma <= 0:
#                 break
#     # while canny_labels.max() > 148:
#     #     sigma += 0.5
#     #     canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)

#     if canny_labels.max() < 4:
#         sigma -= 0.8
#         canny_labels = measure.label(~canny_connect(feature.canny(h, sigma), 7), connectivity=1)

#     if remove_edges:
#         canny_labels = labels_remove_edges(canny_labels)
#     return canny_labels


def surface_overlap(image_labels, canny_labels):
    overlaps = np.zeros(np.unique(image_labels).shape)
    overlaps = []
    joins = 0
    total = 0
    for i in np.unique(image_labels):
        i_labels = canny_labels * 0
        for j in np.unique(canny_labels[image_labels == i]):
            if (image_labels[canny_labels == j] == i).sum() / (canny_labels == j).sum() > 0.5:
                i_labels[canny_labels == j] = 1
        joins += (image_labels[i_labels == 1] == i).sum()
        total += (image_labels == i).sum() + (i_labels == 1).sum()
        rate = 2 * (image_labels[i_labels == 1] == i).sum() / ((image_labels == i).sum() + (i_labels == 1).sum())
        overlaps.append(rate)
    return 2 * joins / total, np.array(overlaps)


def heatmap0(results, ytick=["Multire\nsolution", "Super\nPixels", "MyCanny"], savename="p.png"):
    p = np.array(results)
    xtick = ["Easy", "Medium", "Hard"]
    ax = sns.heatmap(p, cmap="RdYlGn", annot=True, xticklabels=xtick, yticklabels=ytick)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.xaxis.tick_top()
    plt.savefig(savename)
    plt.show()


def heatmap(
    results,
    xtick=["a.简单混合样方", "b.中等混合样方", "c.困难混合样方"],
    ytick=["多尺度分割", "超像素分割", "本文分割"],
    savename="p.png",
):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体为黑体
    p = np.array(results)
    ax = sns.heatmap(p, cmap="RdYlGn", annot=True, xticklabels=xtick, yticklabels=ytick, annot_kws={"fontsize": 20})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.xaxis.tick_bottom

    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=20)

    plt.savefig(savename)
    plt.show()
