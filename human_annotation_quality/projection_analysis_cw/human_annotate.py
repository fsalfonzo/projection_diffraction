import napari
import zarr
from PIL import Image
from glob import glob
import numpy as np

image_dirs = sorted(glob("training_data/*.png"))
images = np.array([np.array(Image.open(x)) for x in image_dirs if "mask" not in x])
if not glob("masks/*"):
    z1 = zarr.open('masks/masks.zarr', mode='w', shape=images.shape,
               chunks=(1,)+images.shape[1:], dtype='uint16')
else:
    z1 = zarr.open("masks/masks.zarr/", mode="r+")
viewer = napari.view_image(images, name='images')
labels_layer = viewer.add_labels(z1, name='segmentation')
napari.run()
