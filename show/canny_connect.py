import numpy as np
from skimage import measure


def canny_connect(im_in, k):
    '''canny算子提取边缘，部分边缘不连通。此函数把边缘连通起来。'''
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
    '''判断是否是边缘结束点，被connect调用。'''
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


if __name__ == '__main__':
    from show import show

    canny_eg = np.zeros((5, 5))
    canny_eg[0:2, 2] = 1
    canny_eg[3:, 2] = 1
    canny_connection = canny_connect(canny_eg, 5)
    print(canny_eg)
    print(canny_connection)

    show([canny_eg, canny_connection], ['canny_eg', 'canny_connection'])
