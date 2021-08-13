import numpy as np
import abc
import os, sys
from pathlib import Path
from PIL import Image

from utils.wavedata.wavedata.tools.core.voxel_grid_2d import VoxelGrid2D
from utils.wavedata.wavedata.tools.core.voxel_grid import VoxelGrid
from utils.wavedata.wavedata.tools.obj_detection import obj_utils
from utils.wavedata.wavedata.tools.core import calib_utils

class BevGenerator:

    @abc.abstractmethod
    def generate_bev(self, **params):
        """Generates BEV maps

        Args:
            **params: additional keyword arguments for
                specific implementations of BevGenerator.

        Returns:
            Dictionary with entries for height maps and one density map
                height_maps: list of height maps
                density_map: density map
        """
        pass

    def _create_density_map(self,
                            num_divisions,
                            voxel_indices_2d,
                            num_pts_per_voxel,
                            norm_value):

        # Create empty density map
        density_map = np.zeros((num_divisions[0],
                                num_divisions[2]))

        # Only update pixels where voxels have num_pts values
        density_map[voxel_indices_2d[:, 0], voxel_indices_2d[:, 1]] = \
            np.minimum(1.0, np.log(num_pts_per_voxel + 1) / norm_value)

        # Density is calculated as min(1.0, log(N+1)/log(x))
        # x=64 for stereo, x=16 for lidar, x=64 for depth
        density_map = np.flip(density_map.transpose(), axis=0)

        return density_map

class Helper(BevGenerator):
    def __init__(self):
        """BEV maps created using slices of the point cloud.
        """

        # Parse config
        self.height_lo = -0.2
        self.height_hi = 2.3
        self.num_slices = 5


        # Pre-calculated values
        self.height_per_division = \
            (self.height_hi - self.height_lo) / self.num_slices

    def create_slice_filter(self, point_cloud, area_extents,
                        ground_plane, ground_offset_dist, offset_dist):
        """ Creates a slice filter to take a slice of the point cloud between
            ground_offset_dist and offset_dist above the ground plane

        Args:
            point_cloud: Point cloud in the shape (3, N)
            area_extents: 3D area extents
            ground_plane: ground plane coefficients
            offset_dist: max distance above the ground
            ground_offset_dist: min distance above the ground plane

        Returns:
            A boolean mask if shape (N,) where
                True indicates the point should be kept
                False indicates the point should be removed
        """

        # Filter points within certain xyz range and offset from ground plane
        offset_filter = obj_utils.get_point_filter(point_cloud, area_extents,
                                                ground_plane, offset_dist)

        # Filter points within 0.2m of the road plane
        road_filter = obj_utils.get_point_filter(point_cloud, area_extents,
                                                ground_plane,
                                                ground_offset_dist)

        slice_filter = np.logical_xor(offset_filter, road_filter)
        return slice_filter


    def create_bev_maps(self,
                        point_cloud,
                        ground_plane,
                        area_extents,
                        voxel_size=0.1):
        """Generates the BEV maps dictionary. One height map is created for
        each slice of the point cloud. One density map is created for
        the whole point cloud.

        Args:
            source: point cloud source 'lidar'
            point_cloud: point cloud (3, N)
            ground_plane: ground plane coefficients
            area_extents: 3D area extents
                [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
            voxel_size: voxel size in m

        Returns:
            BEV maps dictionary
                height_maps: list of height maps
                density_map: density map
        """
        all_points = np.transpose(point_cloud)

        height_maps = []

        for slice_idx in range(self.num_slices):

            height_lo = self.height_lo + slice_idx * self.height_per_division
            height_hi = height_lo + self.height_per_division

            slice_filter = self.create_slice_filter(
                point_cloud,
                area_extents,
                ground_plane,
                height_lo,
                height_hi)

            # Apply slice filter
            slice_points = all_points[slice_filter]

            if len(slice_points) > 1:

                # Create Voxel Grid 2D
                voxel_grid_2d = VoxelGrid2D()
                voxel_grid_2d.voxelize_2d(
                    slice_points, voxel_size,
                    extents=area_extents,
                    ground_plane=ground_plane,
                    create_leaf_layout=False)

                # Remove y values (all 0)
                voxel_indices = voxel_grid_2d.voxel_indices[:, [0, 2]]

            # Create empty BEV images
            height_map = np.zeros((voxel_grid_2d.num_divisions[0],
                                   voxel_grid_2d.num_divisions[2]))

            # Only update pixels where voxels have max height values,
            # and normalize by height of slices
            voxel_grid_2d.heights = voxel_grid_2d.heights - height_lo
            height_map[voxel_indices[:, 0], voxel_indices[:, 1]] = \
                np.asarray(voxel_grid_2d.heights) / self.height_per_division

            height_maps.append(height_map)

        # Rotate height maps 90 degrees
        # (transpose and flip) is faster than np.rot90
        height_maps_out = [np.flip(height_maps[map_idx].transpose(), axis=0)
                           for map_idx in range(len(height_maps))]

        density_slice_filter = self.create_slice_filter(
            point_cloud,
            area_extents,
            ground_plane,
            self.height_lo,
            self.height_hi)

        density_points = all_points[density_slice_filter]

        # Create Voxel Grid 2D
        density_voxel_grid_2d = VoxelGrid2D()
        density_voxel_grid_2d.voxelize_2d(
            density_points,
            voxel_size,
            extents=area_extents,
            ground_plane=ground_plane,
            create_leaf_layout=False)

        # Generate density map
        density_voxel_indices_2d = \
            density_voxel_grid_2d.voxel_indices[:, [0, 2]]

        density_map = self._create_density_map(
            num_divisions=density_voxel_grid_2d.num_divisions,
            voxel_indices_2d=density_voxel_indices_2d,
            num_pts_per_voxel=density_voxel_grid_2d.num_pts_in_voxel,
            norm_value=np.log(16))

        bev_maps = dict()
        bev_maps['height_maps'] = height_maps_out
        bev_maps['density_map'] = density_map

        return bev_maps

    def get_ground_plane(self, sample_name, planes_dir):
        return obj_utils.get_road_plane(int(sample_name), planes_dir)

    def get_calib(self, sample_name, calib_dir):
        return calib_utils.read_calibration(calib_dir, int(sample_name)).p2
    
    def get_point_cloud(self, sample_name, calib_dir, velo_dir, im_size):
        return obj_utils.get_lidar_point_cloud(int(sample_name), calib_dir, velo_dir, im_size=im_size)


"""
Generates 3D anchors, placing them on the ground plane
"""
def generate_anchors_3d(area_extents,
                        anchor_3d_sizes,
                        anchor_stride,
                        ground_plane):
    """
    Tiles anchors over the area extents by using meshgrids to
    generate combinations of (x, y, z), (l, w, h) and ry.

    Args:
        area_extents: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        anchor_3d_sizes: list of 3d anchor sizes N x (l, w, h)
        anchor_stride: stride lengths (x_stride, z_stride)
        ground_plane: coefficients of the ground plane e.g. [0, -1, 0, 0]

    Returns:
        boxes: list of 3D anchors in box_3d format N x [x, y, z, l, w, h, ry]
    """
    # Convert sizes to ndarray
    anchor_3d_sizes = np.asarray(anchor_3d_sizes)

    anchor_stride_x = anchor_stride[0]
    anchor_stride_z = anchor_stride[1]
    anchor_rotations = np.asarray([0, np.pi / 2.0])

    x_start = area_extents[0][0] + anchor_stride[0] / 2.0
    x_end = area_extents[0][1]
    x_centers = np.array(np.arange(x_start, x_end, step=anchor_stride_x),
                         dtype=np.float32)

    z_start = area_extents[2][1] - anchor_stride[1] / 2.0
    z_end = area_extents[2][0]
    z_centers = np.array(np.arange(z_start, z_end, step=-anchor_stride_z),
                         dtype=np.float32)

    # Use ranges for substitution
    size_indices = np.arange(0, len(anchor_3d_sizes))
    rotation_indices = np.arange(0, len(anchor_rotations))

    # Generate matrix for substitution
    # e.g. for two sizes and two rotations
    # [[x0, z0, 0, 0], [x0, z0, 0, 1], [x0, z0, 1, 0], [x0, z0, 1, 1],
    #  [x1, z0, 0, 0], [x1, z0, 0, 1], [x1, z0, 1, 0], [x1, z0, 1, 1], ...]
    before_sub = np.stack(np.meshgrid(x_centers,
                                      z_centers,
                                      size_indices,
                                      rotation_indices),
                          axis=4).reshape(-1, 4)

    # Place anchors on the ground plane
    a, b, c, d = ground_plane
    all_x = before_sub[:, 0]
    all_z = before_sub[:, 1]
    all_y = -(a * all_x + c * all_z + d) / b

    # Create empty matrix to return
    num_anchors = len(before_sub)
    all_anchor_boxes_3d = np.zeros((num_anchors, 7))

    # Fill in x, y, z
    all_anchor_boxes_3d[:, 0:3] = np.stack((all_x, all_y, all_z), axis=1)

    # Fill in shapes
    sizes = anchor_3d_sizes[np.asarray(before_sub[:, 2], np.int32)]
    all_anchor_boxes_3d[:, 3:6] = sizes

    # Fill in rotations
    rotations = anchor_rotations[np.asarray(before_sub[:, 3], np.int32)]
    all_anchor_boxes_3d[:, 6] = rotations

    return all_anchor_boxes_3d

def box_3d_to_anchor(boxes_3d, ortho_rotate=False):
    """ Converts a box_3d [x, y, z, l, w, h, ry]
    into anchor form [x, y, z, dim_x, dim_y, dim_z]

    Anchors in box_3d format should have an ry of 0 or 90 degrees.
    l and w will be matched to dim_x or dim_z depending on the rotation,
    while h will always correspond to dim_y

    Args:
        boxes_3d: N x 7 ndarray of box_3d
        ortho_rotate: optional, if True the box is rotated to the
            nearest 90 degree angle, or else the box is projected
            onto the x and z axes

    Returns:
        N x 6 ndarray of anchors in 'anchor' form
    """

    boxes_3d = np.asarray(boxes_3d).reshape(-1, 7)

    num_anchors = len(boxes_3d)
    anchors = np.zeros((num_anchors, 6))

    # Set x, y, z
    anchors[:, [0, 1, 2]] = boxes_3d[:, [0, 1, 2]]

    # Dimensions along x, y, z
    box_l = boxes_3d[:, [3]]
    box_w = boxes_3d[:, [4]]
    box_h = boxes_3d[:, [5]]
    box_ry = boxes_3d[:, [6]]

    # Rotate to nearest multiple of 90 degrees
    if ortho_rotate:
        half_pi = np.pi / 2
        box_ry = np.round(box_ry / half_pi) * half_pi

    cos_ry = np.abs(np.cos(box_ry))
    sin_ry = np.abs(np.sin(box_ry))

    # dim_x, dim_y, dim_z
    anchors[:, [3]] = box_l * cos_ry + box_w * sin_ry
    anchors[:, [4]] = box_h
    anchors[:, [5]] = box_w * cos_ry + box_l * sin_ry

    return anchors

def anchors_to_box_3d(anchors, fix_lw=False):
    """Converts an anchor form [x, y, z, dim_x, dim_y, dim_z]
    to 3d box format of [x, y, z, l, w, h, ry]

    Note: In this conversion, if the flag 'fix_lw' is set to true,
    the box_3d 'length' will be the longer of dim_x and dim_z, and 'width'
    will be the shorter dimension. All ry values are set to 0.

    Args:
        anchors: N x 6 ndarray of anchors in 'anchor' form
        fix_lw: A boolean flag to switch width and length in the case
            where width is longer than length.

    Returns:
        N x 7 ndarray of box_3d
    """
    anchors = np.asarray(anchors)
    box_3d = np.zeros((len(anchors), 7))

    # Set x, y, z
    box_3d[:, 0:3] = anchors[:, 0:3]
    # Set length to dim_x
    box_3d[:, 3] = anchors[:, 3]
    # Set width to dim_z
    box_3d[:, 4] = anchors[:, 5]
    # Set height to dim_y
    box_3d[:, 5] = anchors[:, 4]
    box_3d[:, 6] = 0

    if fix_lw:
        swapped_indices = box_3d[:, 4] > box_3d[:, 3]
        modified_box_3d = np.copy(box_3d)
        modified_box_3d[swapped_indices, 3] = box_3d[swapped_indices, 4]
        modified_box_3d[swapped_indices, 4] = box_3d[swapped_indices, 3]
        modified_box_3d[swapped_indices, 6] = -np.pi/2
        return modified_box_3d

    return box_3d

def project_to_bev(anchors, bev_extents):
    """
    Projects an array of 3D anchors into bird's eye view

    Args:
        anchors: list of anchors in anchor format (N x 6):
            N x [x, y, z, dim_x, dim_y, dim_z],
            can be a numpy array or tensor
        bev_extents: xz extents of the 3d area
            [[min_x, max_x], [min_z, max_z]]

    Returns:
          box_corners_norm: corners as a percentage of the map size, in the
            format N x [x1, y1, x2, y2]. Origin is the top left corner
    """

    anchors = np.asarray(anchors)

    x = anchors[:, 0]
    z = anchors[:, 2]
    half_dim_x = anchors[:, 3] / 2.0
    half_dim_z = anchors[:, 5] / 2.0

    # Calculate extent ranges
    bev_x_extents_min = bev_extents[0][0]
    bev_z_extents_min = bev_extents[1][0]
    bev_x_extents_max = bev_extents[0][1]
    bev_z_extents_max = bev_extents[1][1]
    bev_x_extents_range = bev_x_extents_max - bev_x_extents_min
    bev_z_extents_range = bev_z_extents_max - bev_z_extents_min

    # 2D corners (top left, bottom right)
    x1 = x - half_dim_x
    x2 = x + half_dim_x
    # Flip z co-ordinates (origin changes from bottom left to top left)
    z1 = bev_z_extents_max - (z + half_dim_z)
    z2 = bev_z_extents_max - (z - half_dim_z)

    # Stack into (N x 4)
    bev_box_corners = np.stack([x1, z1, x2, z2], axis=1)

    # Convert from original xz into bev xz, origin moves to top left
    bev_extents_min_tiled = [bev_x_extents_min, bev_z_extents_min,
                             bev_x_extents_min, bev_z_extents_min]
    bev_box_corners = bev_box_corners - bev_extents_min_tiled

    # Calculate normalized box corners for ROI pooling
    extents_tiled = [bev_x_extents_range, bev_z_extents_range,
                     bev_x_extents_range, bev_z_extents_range]
    bev_box_corners_norm = bev_box_corners / extents_tiled

    return bev_box_corners, bev_box_corners_norm


def project_to_image_space(anchors, stereo_calib_p2, image_shape):
    """
    Projects 3D anchors into image space

    Args:
        anchors: list of anchors in anchor format N x [x, y, z,
            dim_x, dim_y, dim_z]
        stereo_calib_p2: stereo camera calibration p2 matrix
        image_shape: dimensions of the image [h, w]

    Returns:
        box_corners: corners in image space - N x [x1, y1, x2, y2]
        box_corners_norm: corners as a percentage of the image size -
            N x [x1, y1, x2, y2]
    """
    if anchors.shape[1] != 6:
        raise ValueError("Invalid shape for anchors {}, should be "
                         "(N, 6)".format(anchors.shape[1]))

    # Figure out box mins and maxes
    x = (anchors[:, 0])
    y = (anchors[:, 1])
    z = (anchors[:, 2])

    dim_x = (anchors[:, 3])
    dim_y = (anchors[:, 4])
    dim_z = (anchors[:, 5])

    dim_x_half = dim_x / 2.
    dim_z_half = dim_z / 2.

    # Calculate 3D BB corners
    x_corners = np.array([x + dim_x_half,
                          x + dim_x_half,
                          x - dim_x_half,
                          x - dim_x_half,
                          x + dim_x_half,
                          x + dim_x_half,
                          x - dim_x_half,
                          x - dim_x_half]).T.reshape(1, -1)

    y_corners = np.array([y,
                          y,
                          y,
                          y,
                          y - dim_y,
                          y - dim_y,
                          y - dim_y,
                          y - dim_y]).T.reshape(1, -1)

    z_corners = np.array([z + dim_z_half,
                          z - dim_z_half,
                          z - dim_z_half,
                          z + dim_z_half,
                          z + dim_z_half,
                          z - dim_z_half,
                          z - dim_z_half,
                          z + dim_z_half]).T.reshape(1, -1)

    anchor_corners = np.vstack([x_corners, y_corners, z_corners])

    # Apply the 2D image plane transformation
    pts_2d = calib_utils.project_to_image(anchor_corners, stereo_calib_p2)

    # Get the min and maxes of image coordinates
    i_axis_min_points = np.amin(pts_2d[0, :].reshape(-1, 8), axis=1)
    j_axis_min_points = np.amin(pts_2d[1, :].reshape(-1, 8), axis=1)

    i_axis_max_points = np.amax(pts_2d[0, :].reshape(-1, 8), axis=1)
    j_axis_max_points = np.amax(pts_2d[1, :].reshape(-1, 8), axis=1)

    box_corners = np.vstack([i_axis_min_points, j_axis_min_points,
                             i_axis_max_points, j_axis_max_points]).T

    # Normalize
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    image_shape_tiled = [image_shape_w, image_shape_h,
                         image_shape_w, image_shape_h]

    box_corners_norm = box_corners / image_shape_tiled

    return np.array(box_corners, dtype=np.float32), \
        np.array(box_corners_norm, dtype=np.float32)

def box_3d_to_object_label(box_3d, obj_type='Car'):
    """Turns a box_3d into an ObjectLabel

    Args:
        box_3d: 3D box in the format [x, y, z, l, w, h, ry]
        obj_type: Optional, the object type

    Returns:
        ObjectLabel with the location, size, and rotation filled out
    """

    obj_label = obj_utils.ObjectLabel()

    obj_label.type = obj_type

    obj_label.t = box_3d.take((0, 1, 2))
    obj_label.l = box_3d[3]
    obj_label.w = box_3d[4]
    obj_label.h = box_3d[5]
    obj_label.ry = box_3d[6]

    return obj_label

def project_to_image_space_box3d(box_3d, calib_p2,
                           truncate=False, image_size=None,
                           discard_before_truncation=True):
    """ Projects a box_3d into image space

    Args:
        box_3d: single box_3d to project
        calib_p2: stereo calibration p2 matrix
        truncate: if True, 2D projections are truncated to be inside the image
        image_size: [w, h] must be provided if truncate is True,
            used for truncation
        discard_before_truncation: If True, discard boxes that are larger than
            80% of the image in width OR height BEFORE truncation. If False,
            discard boxes that are larger than 80% of the width AND
            height AFTER truncation.

    Returns:
        Projected box in image space [x1, y1, x2, y2]
            Returns None if box is not inside the image
    """

    obj_label = box_3d_to_object_label(box_3d)
    corners_3d = obj_utils.compute_box_corners_3d(obj_label)

    projected = calib_utils.project_to_image(corners_3d, calib_p2)

    x1 = np.amin(projected[0])
    y1 = np.amin(projected[1])
    x2 = np.amax(projected[0])
    y2 = np.amax(projected[1])

    img_box = np.array([x1, y1, x2, y2])

    if truncate:
        if not image_size:
            raise ValueError('Image size must be provided')

        image_w = image_size[0]
        image_h = image_size[1]

        # Discard invalid boxes (outside image space)
        if img_box[0] > image_w or \
                img_box[1] > image_h or \
                img_box[2] < 0 or \
                img_box[3] < 0:
            return None

        # Discard boxes that are larger than 80% of the image width OR height
        if discard_before_truncation:
            img_box_w = img_box[2] - img_box[0]
            img_box_h = img_box[3] - img_box[1]
            if img_box_w > (image_w * 0.8) or img_box_h > (image_h * 0.8):
                return None

        # Truncate remaining boxes into image space
        if img_box[0] < 0:
            img_box[0] = 0
        if img_box[1] < 0:
            img_box[1] = 0
        if img_box[2] > image_w:
            img_box[2] = image_w
        if img_box[3] > image_h:
            img_box[3] = image_h

        # Discard boxes that are covering the the whole image after truncation
        if not discard_before_truncation:
            img_box_w = img_box[2] - img_box[0]
            img_box_h = img_box[3] - img_box[1]
            if img_box_w > (image_w * 0.8) and img_box_h > (image_h * 0.8):
                return None

    return img_box


def save_predictions_in_kitti_format(sample_list,
                                     score_threshold):
    """ Converts a set of network predictions into text files required for
    KITTI evaluation.
    """
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    img_dir = ROOT_DIR / 'data/image_2'
    calib_dir = ROOT_DIR / 'data/calib'
    helper = Helper()
    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    # Get available prediction folders
    predictions_root_dir = ROOT_DIR / "output"

    final_predictions_dir = predictions_root_dir / 'final_predictions_and_scores'

    # 3D prediction directories
    kitti_predictions_3d_dir = ROOT_DIR / 'kitti_native_eval' / str(score_threshold) / 'data'

    kitti_predictions_3d_dir.mkdir(parents=True, exist_ok=True)

    # Do conversion
    num_samples = len(sample_list)
    num_valid_samples = 0

    print('Converting detections from:', final_predictions_dir)

    print('3D Detections being saved to:', kitti_predictions_3d_dir)

    for idx, sample_name in enumerate(sample_list):
        # sample_name = "000002"
        # Print progress
        sys.stdout.write('\rConverting {} / {}'.format(
            idx + 1, num_samples))
        sys.stdout.flush()

        prediction_file = sample_name + '.txt'

        kitti_predictions_3d_file_path = kitti_predictions_3d_dir / prediction_file

        predictions_file_path = final_predictions_dir / prediction_file

        # If no predictions, skip to next file
        if not os.path.exists(predictions_file_path):
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        all_predictions = np.loadtxt(predictions_file_path)

        # # Swap l, w for predictions where w > l
        # swapped_indices = all_predictions[:, 4] > all_predictions[:, 3]
        # fixed_predictions = np.copy(all_predictions)
        # fixed_predictions[swapped_indices, 3] = all_predictions[
        #     swapped_indices, 4]
        # fixed_predictions[swapped_indices, 4] = all_predictions[
        #     swapped_indices, 3]
        # print()
        # np.set_printoptions(threshold=np.inf)
        # np.set_printoptions(suppress=True)
        # tmp = np.around(all_predictions, 3)
        # print(tmp); raise Exception
        score_filter = all_predictions[:, 7] >= score_threshold
        all_predictions = all_predictions[score_filter]

        # If no predictions, skip to next file
        if len(all_predictions) == 0:
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        # Project to image space
        sample_name = prediction_file.split('.')[0]
        img_idx = int(sample_name)

        # Load image for truncation
        img_path = str(img_dir / (sample_name+'.png'))
        image = Image.open(img_path)

        stereo_calib_p2 = helper.get_calib(sample_name, str(calib_dir))

        boxes = []
        image_filter = []
        for i in range(len(all_predictions)):
            box_3d = all_predictions[i, 0:7]
            img_box = project_to_image_space_box3d(
                box_3d, stereo_calib_p2,
                truncate=True, image_size=image.size)

            # Skip invalid boxes (outside image space)
            if img_box is None:
                image_filter.append(False)
                continue

            image_filter.append(True)
            boxes.append(img_box)

        boxes = np.asarray(boxes)
        all_predictions = all_predictions[image_filter]

        # If no predictions, skip to next file
        if len(boxes) == 0:
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        num_valid_samples += 1

        # To keep each value in its appropriate position, an array of zeros
        # (N, 16) is allocated but only values [4:16] are used
        kitti_predictions = np.zeros([len(boxes), 16])

        # Get object types
        all_pred_classes = all_predictions[:, 8].astype(np.int32)
        obj_types = ['Car' if class_idx == 0 else f'People_{class_idx}' 
                     for class_idx in all_pred_classes]

        # Truncation and Occlusion are always empty (see below)

        # Alpha (Not computed)
        kitti_predictions[:, 3] = -10 * np.ones((len(kitti_predictions)),
                                                dtype=np.int32)

        # 2D predictions
        kitti_predictions[:, 4:8] = boxes[:, 0:4]

        # 3D predictions
        # (l, w, h)
        kitti_predictions[:, 8] = all_predictions[:, 5]
        kitti_predictions[:, 9] = all_predictions[:, 4]
        kitti_predictions[:, 10] = all_predictions[:, 3]
        # (x, y, z)
        kitti_predictions[:, 11:14] = all_predictions[:, 0:3]
        # (ry, score)
        kitti_predictions[:, 14:16] = all_predictions[:, 6:8]

        # Round detections to 3 decimal places
        kitti_predictions = np.round(kitti_predictions, 3)

        # Empty Truncation, Occlusion
        kitti_empty_1 = -1 * np.ones((len(kitti_predictions), 2),
                                     dtype=np.int32)

        # Stack 3D predictions text
        kitti_text_3d = np.column_stack([obj_types,
                                         kitti_empty_1,
                                         kitti_predictions[:, 3:16]])
        # print("\n", sample_name)
        # print(kitti_text_3d); raise Exception
        # Save to text files
        np.savetxt(kitti_predictions_3d_file_path, kitti_text_3d,
                   newline='\r\n', fmt='%s')

    print('\nNum valid:', num_valid_samples)
    print('Num samples:', num_samples)
