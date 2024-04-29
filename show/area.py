'''计算影像单个像素点的实际面积。'''
import rasterio
from geographiclib.geodesic import Geodesic
from geopy.distance import distance  # distance等价于geodesic


def pixel_area(file):
    '''计算影像file单个像素点的实际面积。'''
    im = rasterio.open(file)
    t = im.transform
    pixel_width = (
        Geodesic.WGS84.Inverse((t * (0, 0))[1], (t * (0, 0))[0], (t * (im.width, 0))[1], (t * (im.width, 0))[0])['s12']
        / im.width
    )
    pixel_height = (
        Geodesic.WGS84.Inverse((t * (0, 0))[1], (t * (0, 0))[0], (t * (0, im.height))[1], (t * (0, im.height))[0])[
            's12'
        ]
        / im.height
    )
    return pixel_width * pixel_height


def pixel_area2(file):
    '''计算影像file单个像素点的实际面积。'''
    im = rasterio.open(file)
    t = im.transform
    #               model             major (km)   minor (km)     flattening
    # ELLIPSOIDS = {'WGS-84':        (6378.137,    6356.7523142,  1 / 298.257223563),
    #               'GRS-80':        (6378.137,    6356.7523141,  1 / 298.257222101),
    #               'Airy (1830)':   (6377.563396, 6356.256909,   1 / 299.3249646),
    #               'Intl 1924':     (6378.388,    6356.911946,   1 / 297.0),
    #               'Clarke (1880)': (6378.249145, 6356.51486955, 1 / 293.465),
    #               'GRS-67':        (6378.1600,   6356.774719,   1 / 298.25),
    #               }
    # eg = distance(list(t*(0,0))[::-1], list(t*(im.width, 0))[::-1], ellipsoid='WGS-84').m
    pixel_width = distance(list(t * (0, 0))[::-1], list(t * (im.width, 0))[::-1]).m / im.width
    pixel_height = distance(list(t * (0, 0))[::-1], list(t * (0, im.height))[::-1]).m / im.height
    return pixel_width * pixel_height


if __name__ == '__main__':
    file = 'image_smallest.tif'
    print(pixel_area(file))
    print(pixel_area2(file))
