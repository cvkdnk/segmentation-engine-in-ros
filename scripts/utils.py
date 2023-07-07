import os
import numpy as np


def load_data(bin_path: str, return_ins_label=False, test_mode=False):
    pt_features = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    if test_mode:
        sem_labels = np.zeros(pt_features.shape[0], dtype=np.int32)
        return pt_features, None
    labels = np.fromfile(bin_path.replace("velodyne", "labels")[:-3]+"label", np.uint32)
    sem_labels = labels & 0xFFFF
    ins_labels = None
    frame = os.path.splitext(os.path.basename(bin_path))[0]
    seq = os.path.dirname(os.path.dirname(bin_path))[-2:]
    seq_frame = seq+"_"+frame
    if return_ins_label:
        ins_labels = labels >> 16
    if return_ins_label:
        return pt_features, sem_labels, ins_labels, seq_frame
    else:
        return pt_features, sem_labels, ins_labels, seq_frame


def label_mapping(labels, label_map):
    return np.vectorize(label_map.__getitem__)(labels)


def label2word(labels, word_mapping, learning_map_inv=None):
    """If learning_map_inv is not None, it should give a dict."""
    map_labels = np.copy(labels)
    if learning_map_inv is not None:
        map_labels = np.vectorize(learning_map_inv.__getitem__)(labels)
    words = np.vectorize(word_mapping.__getitem__)(map_labels)
    return words


def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)
    phi = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    return np.hstack((rho.reshape(-1, 1), phi.reshape(-1, 1), input_xyz[..., 2:]))


def polar2cart(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[..., 0] * np.cos(input_xyz_polar[..., 1])
    y = input_xyz_polar[..., 0] * np.sin(input_xyz_polar[..., 1])
    return np.stack((x, y, input_xyz_polar[..., 2:]), axis=-1)


# def cart2polar3d(input_xyz):
#     rho = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)
#     phi = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
#     theta = np.arctan2(input_xyz[..., 2], rho)
#     return np.stack((rho, phi, theta), axis=-1)
#
#
# def polar2cart3d(input_xyz):
#     x = input_xyz[..., 0] * np.cos(input_xyz[..., 1]) * np.sin(input_xyz[..., 2])
#     y = input_xyz[..., 0] * np.sin(input_xyz[..., 1]) * np.sin(input_xyz[..., 2])
#     z = input_xyz[..., 0] * np.cos(input_xyz[..., 2])
#     return np.stack((x, y, z), axis=-1)


def cart2spherical(input_xyz):
    """return rho, phi, theta; also known as depth, yaw, pitch"""
    depth = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2 + input_xyz[..., 2] ** 2)
    yaw = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    pitch = np.arcsin(input_xyz[..., 2] / depth)
    return np.stack((depth, yaw, pitch), axis=-1)


def spherical2cart(input_dyp):
    x = input_dyp[..., 0] * np.cos(input_dyp[..., 2]) * np.cos(input_dyp[..., 1])
    y = input_dyp[..., 0] * np.cos(input_dyp[..., 2]) * np.sin(input_dyp[..., 1])
    z = input_dyp[..., 0] * np.sin(input_dyp[..., 2])
    return np.stack((x, y, z), axis=-1)


class RangeProject():
    def __init__(self, proj_H=64, proj_W=1024, proj_fov_up=3, proj_fov_down=-25):
        super(RangeProject, self).__init__()
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.proj_fov_up = proj_fov_up / 180.0 * np.pi
        self.proj_fov_down = proj_fov_down / 180.0 * np.pi

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
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # mapping to range image and inverse mapping
        r2p = np.zeros((pt_features.shape[0], 2), dtype=np.int32)
        p2r = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        r2p[..., 0] = proj_y
        r2p[..., 1] = proj_x
        p2r[proj_y, proj_x] = indices

        return {"Range": {
            "r2p": r2p,
            "p2r": p2r,
        }}



def color_map(raw_labels, map_dict):
    color_fea = np.empty((raw_labels.shape[0], 3), dtype=np.int32)
    for i, label in enumerate(raw_labels):
        color_fea[i] = map_dict[label]
    return color_fea