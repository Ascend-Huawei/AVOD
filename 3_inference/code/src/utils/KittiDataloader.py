import os, sys
import numpy as np
import cv2
from pathlib import Path

import utils.utils as utils # Helper, generate_anchors_3d, box_3d_to_anchor

class KittiDataloader:
    def __init__(self, ):
        
        self.ROOT_DIR = Path(__file__).resolve().parent.parent.parent

        self.dataset_dir = self.ROOT_DIR / 'data'

        self.imgs_dir = self.dataset_dir / 'image_2'
        self.calib_dir = self.dataset_dir / 'calib'
        self.planes_dir = self.dataset_dir / 'planes'
        self.velo_dir = self.dataset_dir / 'velodyne'

        self.sample_names = [name for name, _ in map(lambda x : x.split('.'), os.listdir(self.imgs_dir))]
        self.idx = 0
        self.num_samples = len(self.sample_names)
        
        self.img_size = (1200, 360)

        self.area_extents = np.reshape([-40, 40, -5, 3, 0, 70], (3, 2))
        self.anchor_3d_sizes = [np.array([3.514, 1.581, 1.511]), np.array([4.236, 1.653, 1.547])]
        self.anchor_stride = np.array([0.5, 0.5])

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_samples:
            raise StopIteration

        sys.stdout.write('\rInferencing {} / {}\n'.format(
            self.idx + 1, self.num_samples))
        sys.stdout.flush()
        
        name = self.sample_names[self.idx]
        self.idx += 1

        img = cv2.imread(str(self.imgs_dir  / (name+'.png')))[..., :: -1]
        tmp_img_size = (img.shape[1], img.shape[0])
        img = cv2.resize(img, self.img_size).astype(np.float32)
        helper = utils.Helper()
        calib = helper.get_calib(name, str(self.calib_dir))
        ground_plane = helper.get_ground_plane(name, str(self.planes_dir))
        point_cloud = helper.get_point_cloud(name, str(self.calib_dir), str(self.velo_dir), tmp_img_size)

        bev_maps = helper.create_bev_maps(point_cloud, ground_plane, self.area_extents)
        height_maps = bev_maps.get('height_maps')
        density_map = bev_maps.get('density_map')

        bev_input = np.dstack((*height_maps, density_map))
        bev_input = np.pad(bev_input, ((4,0), (0, 0), (0, 0)))

        all_anchor_boxes_3d = utils.generate_anchors_3d(self.area_extents, self.anchor_3d_sizes, self.anchor_stride, ground_plane)
        anchors_to_use = utils.box_3d_to_anchor(all_anchor_boxes_3d)

        _, bev_anchors_norm = utils.project_to_bev(anchors_to_use, np.array([[-40.,  40.], [  0.,  70.]]))
        _, img_anchors_norm = utils.project_to_image_space(anchors_to_use, calib, [360, 1200])

        # Reorder into [y1, x1, y2, x2] for tf.crop_and_resize op
        bev_anchors_norm = bev_anchors_norm[:, [1, 0, 3, 2]]
        img_anchors_norm = img_anchors_norm[:, [1, 0, 3, 2]]
        
        res = [bev_input, img, anchors_to_use, bev_anchors_norm, img_anchors_norm, calib, ground_plane]
        
        return name, list(map(lambda x : x.astype(np.float32).copy(), res))

dataloader = KittiDataloader()
dataloader.__next__()