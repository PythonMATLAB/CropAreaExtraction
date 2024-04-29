"""转换影像投影"""
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def image_reproject(image_name):
    """转换影像投影"""
    dst_crs = 'EPSG:32612'
    with rasterio.open(image_name) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
        dst_name = image_name.split('.')[0] + '_reproject.' + image_name.split('.')[1]
        with rasterio.open(dst_name, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )


if __name__ == '__main__':
    image_name = 'image_smallest.tif'
    image_reproject(image_name)
    left = -4959823.301465246
    bottom = 15869316.733877186
    right = -4959782.016831959
    top = 15869353.231886324
