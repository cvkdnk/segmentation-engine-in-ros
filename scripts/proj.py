import numpy as np


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

        return {"Range": {
            "range_image": range_image,
            "range_mask": range_mask,
            "r2p": r2p,
            "p2r": p2r,
        }}


def label_mapping(labels, label_map):
    return np.vectorize(label_map.__getitem__)(labels)


if __name__ == "__main__":
    range_proj = RangeProject(proj_H=16, proj_W=1024, proj_fov_up=3, proj_fov_down=-25)
