
from Im_Nod_info import *
import os
import numpy as np
from skimage import io, draw


lst = open('../data/csv/train/annotations.csv').readlines()

for i in lst[1:]:
	uid,coordX,coordY,coordZ,diameter_mm = i.split('\r\n')[0].split(',')

	image_path = '../../data_lung/train/{}.mhd'.format(uid)
	save_path = '../results/overviews/{}_{}.jpg'.format(uid,diameter_mm)

	if not os.path.exists(image_path):
		continue

	world_coord = np.array([coordX,coordY,coordZ], dtype=np.float32)
	
	node = Nodules(world_coord)
	node.diam = diameter_mm
	scans = Image_CT_Scans(image_path)
	voxel_coord = node.world_to_voxel(scans.origin, scans.spacing)

	image = scans.image_normalization()[voxel_coord[2]]
	image = get_segmented_lungs(image)
	dian = np.array(np.array(diameter_mm,dtype=np.float)/scans.spacing[0], dtype=np.int)+3
	rr, cc = draw.circle_perimeter(voxel_coord[1], voxel_coord[0], dian)
	image[rr, cc] = 1
	io.imsave(save_path, image)


