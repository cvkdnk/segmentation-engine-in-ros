import numpy as np
import logging
import time


logger = logging.getLogger("proj")


def cart2spherical(input_xyz):
    """return rho, phi, theta; also known as depth, yaw, pitch"""
    depth = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2 + input_xyz[..., 2] ** 2)
    yaw = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    pitch = np.arcsin(input_xyz[..., 2] / depth)
    return np.stack((depth, yaw, pitch), axis=-1)


class RangeProject():
    RETURN_TYPE = "Range"

    def __init__(self, **config):
        super(RangeProject, self).__init__()
        self.proj_H = config["proj_H"]
        self.proj_W = config["proj_W"]
        self.proj_fov_up = config["proj_fov_up"] / 180.0 * np.pi
        self.proj_fov_down = config["proj_fov_down"] / 180.0 * np.pi

    @classmethod
    def gen_config_template(cls):
        return {
            "proj_H": 64,
            "proj_W": 1024,
            "proj_fov_up": 3,
            "proj_fov_down": -25
        }

    def __call__(self, data):
        logger.debug("Range Project "+("-"*20))
        pt_features = data["Point"]
        coords = pt_features[..., :3]
        coords_sph = cart2spherical(coords)
        coords_sph[..., 2] = np.clip(coords_sph[..., 2], self.proj_fov_down, self.proj_fov_up)
        fov = self.proj_fov_up - self.proj_fov_down
        depth = coords_sph[..., 0]
        yaw = coords_sph[..., 1]
        pitch = coords_sph[..., 2]

        # project to image
        proj_x = 0.5 * (yaw / np.pi + 1.0) * self.proj_W
        proj_y = (1.0 - (pitch - self.proj_fov_down) / fov) * self.proj_H

        # round and clamp for use as index
        proj_x = np.floor(proj_x).astype(np.int32)
        proj_x = np.clip(proj_x, 0, self.proj_W - 1)

        proj_y = np.floor(proj_y).astype(np.int32)
        proj_y = np.clip(proj_y, 0, self.proj_H - 1)

        unproj_range = np.copy(depth)
        # order in decreasing depth
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = order
        order_pt_features = pt_features[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # mapping to range image and inverse mapping
        r2p = np.zeros((pt_features.shape[0], 2), dtype=np.int32)
        p2r = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        r2p[..., 0] = proj_y
        r2p[..., 1] = proj_x
        p2r[proj_y, proj_x] = indices
        range_image = np.full((self.proj_H, self.proj_W, 5), -1, dtype=np.float32)
        range_image[proj_y, proj_x, 0] = depth
        range_image[proj_y, proj_x, 1:] = order_pt_features
        range_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        range_mask[proj_y, proj_x] = 1
        logger.debug("valid pixels count: %d / %d" % (np.sum(range_mask), range_mask.shape[0]+range_mask.shape[1]))

        return {"Range": {
            "range_image": range_image,
            "range_mask": range_mask,
            "r2p": r2p,
            "p2r": p2r,
        }}


class BevProject():
    RETURN_TYPE="Bev"

    def __init__(self, **config):
        self.resolution = config["resolution"]
        self.x_range = config["x_range"]
        self.y_range = config["y_range"]

    @classmethod
    def gen_config_template(cls):
        return {
            "resolution": 0.2,
            "x_range": (-50, 50),
            "y_range": (-50, 50)
        }

            
    def __call__(self, data):
        """
        Convert a point cloud to a bird's eye view image.

        Args:
            points: (N, 3) numpy array, the point cloud, each row is a point (x, y, z).
            self.resolution: float, the self.resolution of the BEV image.
            width_range: tuple of float, (min, max) width values, points outside this range are ignored.
            length_range: tuple of float, (min, max) length values, points outside this range are ignored.

        Returns:
        """
        logger.debug("BEV Project "+("-"*20))
        points = data["Point"]
        
        # filt points by self.x_range and self.y_range
        select_points = (points[:, 0] > self.x_range[0]) & (points[:, 0] < self.x_range[1]) & (points[:, 1] > self.y_range[0]) & (points[:, 1] < self.y_range[1])
        logger.debug("select %d/%d points (by filter)" % (select_points.sum(), points.shape[0]))

        # clip points and make b2p
        clip_points = points.copy()
        clip_points[:, 0] = np.clip(clip_points[:, 0], self.x_range[0], self.x_range[1])
        clip_points[:, 1] = np.clip(clip_points[:, 1], self.y_range[0], self.y_range[1])
        b2p = clip_points[:, :2] / self.resolution
        b2p[:, 0] -= self.x_range[0] / self.resolution
        b2p[:, 1] -= self.y_range[0] / self.resolution

        # get original indices
        select_points_indices = select_points.nonzero()[0]
        selected_points = points[select_points]

        # compute points' pixel coords in image
        pt_img_coords = (selected_points[:, :2] / self.resolution)
        pt_img_coords[:, 0] -= self.x_range[0] / self.resolution
        pt_img_coords[:, 1] -= self.y_range[0] / self.resolution
        pt_img_coords = pt_img_coords.astype(np.int64)

        # process conflict by z_max
        unq, unq_inv, unq_cnt = np.unique(pt_img_coords, return_inverse=True, return_counts=True, axis=0)
        conflict = unq_cnt > 1
        logger.debug("conflict: %d / %d pixels, max %d points conflict" % (conflict.sum(), conflict.shape[0], unq_cnt.max()))
        
        z_values = selected_points[:, 2]
        max_z_indices = np.zeros(unq.shape[0], dtype=np.int64)
        for idx in range(unq.shape[0]):
            indices_in_group = np.where(unq_inv == idx)[0]
            max_z_index_in_group = indices_in_group[np.argmax(z_values[indices_in_group])]
            max_z_indices[idx] = max_z_index_in_group
        max_z_coords = pt_img_coords[max_z_indices]

        # gen p2b and mask
        p2b = np.full((int((self.x_range[1]-self.x_range[0])/self.resolution), int((self.y_range[1]-self.y_range[0])/self.resolution)), -1, dtype=np.int64)
        p2b[max_z_coords[:, 0], max_z_coords[:, 1]] = select_points_indices[max_z_indices]
        mask = np.zeros_like(p2b, dtype=np.int64)
        mask[max_z_coords[:, 0], max_z_coords[:, 1]] = 1

        return {"Bev": {
            "p2b": p2b,
            "b2p": pt_img_coords,
            "mask": mask
        }}


def label_mapping(labels, label_map):
    return np.vectorize(label_map.__getitem__)(labels)


if __name__ == "__main__":
    range_proj = RangeProject(proj_H=16, proj_W=1024, proj_fov_up=3, proj_fov_down=-25)
    points = np.array([[0.1,0.1,0.1],[1.1,1.1,1.1],[3.1,1.1,0.5],[0.2,0.3,0.2]])
