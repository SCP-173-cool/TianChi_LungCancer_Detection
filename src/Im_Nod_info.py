

import numpy as np
import SimpleITK as sitk
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.segmentation import clear_border
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi


class Image_CT_Scans(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.sitk_image = sitk.ReadImage(file_name)
        self.origin = np.array(self.sitk_image.GetOrigin())
        self.spacing = np.array(self.sitk_image.GetSpacing())

    def read_ct_image(self):
        image = sitk.GetArrayFromImage(self.sitk_image)
        self.image_shape = image.shape
        return image

    def image_normalization(self, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
        image = self.read_ct_image()
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1
        image[image < 0] = 0
        return image

class Nodules(object):
    def __init__(self, world_coord):
        self.uid = ''
        self.world_coord = world_coord
        self.diam = 0

    def world_to_voxel(self, origin, spacing):

        voxel_coord = np.absolute(self.world_coord - origin) / spacing
        voxel_coord = voxel_coord.astype(np.int32)
        self.voxel_coord = voxel_coord 
        return voxel_coord

def get_segmented_lungs(im, val=0.24):
    image = im
    binary = image < val
    cleared = clear_border(binary)
    label_image = label(cleared)
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    selem = disk(10)
    binary = binary_closing(binary, selem)
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    get_high_vals = binary == 0
    image[get_high_vals] = 0
    return image
