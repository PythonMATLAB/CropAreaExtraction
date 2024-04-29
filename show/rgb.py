'''对比rgb图层'''
import matplotlib.pyplot as plt
from matplotlib.image import imread


def rgb(image_name):
    '''对比rgb图层'''
    # read as a ndarray
    image = imread(image_name)
    # 作图
    fig = plt.figure('show rgb', figsize=(20, 10))
    axs = fig.subplots(2, 2)
    axs = axs.flatten()
    title = ['origin', 'r', 'g', 'b']
    for i in range(4):
        if i == 0:
            axs[i].imshow(image)
        else:
            axs[i].imshow(image[:, :, i - 1], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(title[i])
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    image_name = 'image_smallest.tif'
    rgb(image_name)
