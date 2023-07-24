import argparse
import numpy as np
import open3d as o3d
import glob
import cv2
import yaml
import os
from tqdm import tqdm

from proj import RangeProject, label_mapping


""" dir structure like semantic kitti:
    velodyne/*.bin is points cloud data
    predictions/*.label is predict label data
    images/*.png is range rgb image colored by labels
    depth/*.png is range grey image colored by depth
    xyz/*.xyz is points cloud and rgb data in csv file structure
    mix_video is visualize video by concatenating range imgs"""

# DIR
ROOT_PATH = "/root/autodl-tmp/sequences/11/"
SEMANTICKITTI_YAML_PATH = "/root/autodl-tmp/sequences/11/semantic-kitti.yaml"

# Visualize Images
PROJ_HW = [16, 512]
FOV_UP = 15
FOV_DOWN = -15

# Generate Video
FOURCC = "MJPG"
SCALE = [1, 2]  # W, H
FPS = 5.0


class AverageMeter(object):
    def __init__(self, dim):
        self.dim = dim
        self.reset()

    def reset(self):
        # 初始化计数器，总和，平方和，平均值和标准差
        self.count = 0
        self.sum = np.zeros(self.dim)
        self.sqsum = np.zeros(self.dim)
        self.avg = np.zeros(self.dim)
        self.std = np.zeros(self.dim)

    def update(self, val, n=1):
        assert val.shape[1] == self.dim
        # 更新计数器，总和，平方和
        self.count += n
        self.sum += np.sum(val, axis=0) * n
        self.sqsum += np.sum(np.square(val), axis=0) * n
        # 计算平均值和标准差
        self.avg = self.sum / self.count
        self.std = np.sqrt(self.sqsum / self.count - np.square(self.avg))

    def __str__(self):
        # 返回平均值和标准差的字符串表示
        avg_str = ', '.join([f"{x:.4f}" for x in self.avg])
        std_str = ', '.join([f"{x:.4f}" for x in self.std])
        return f"avg: {avg_str}, std: {std_str}"


def read_label_colors(file_path, swap_bgr=True):
    with open(file_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    color_map = data_loaded['color_map']
    # swap colors from BGR to RGB
    if swap_bgr:
        for key in color_map:
            color_map[key] = color_map[key][::-1]
    return color_map


def cart2spherical(input_xyz):
    """return rho, phi, theta; also known as depth, yaw, pitch"""
    depth = np.sqrt(input_xyz[..., 0]**2 + input_xyz[..., 1]**2 +
                    input_xyz[..., 2]**2)
    yaw = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    pitch = np.arcsin(input_xyz[..., 2] / depth)

    return np.stack((depth, yaw, pitch), axis=-1)


def read_point_cloud_and_labels(pcd_path, label_path, color_map):
    point_cloud = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    sph_coords = cart2spherical(point_cloud)
    labels = np.fromfile(label_path, dtype=np.int32)
    with open(SEMANTICKITTI_YAML_PATH, 'r') as f:
        kitti_config = yaml.safe_load(f)
    # Create an empty list to store RGB colors
    colors = np.empty([len(labels), 3])

    # Color the point cloud based on the labels

    for label_id, color in color_map.items():
        colors[labels == label_id] = color

    return point_cloud, colors


def plotRangeImages(proj_H, proj_W, proj_fov_up, proj_fov_down, show_data_details=False):
    # Load label colors
    color_map = read_label_colors(SEMANTICKITTI_YAML_PATH)

    # Directory paths
    pcd_dir = ROOT_PATH+'velodyne/'
    label_dir = ROOT_PATH+'predictions/'

    # Sorted list of point cloud and label files
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.bin')))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.label')))

    # Store point clouds
    point_clouds = []

    # Create a directory to store images
    os.makedirs(ROOT_PATH+'images', exist_ok=True)
    os.makedirs(ROOT_PATH+'depth', exist_ok=True)
    os.makedirs(ROOT_PATH+'preds', exist_ok=True)

    # Init RangeProject
    proj = RangeProject(proj_H=proj_H, proj_W=proj_W,
                        proj_fov_up=proj_fov_up, proj_fov_down=proj_fov_down)
    
    # Init Meter
    if show_data_details:
        meters = AverageMeter(4)

    for i, (pcd_file, label_file) in tqdm(enumerate(zip(pcd_files, label_files)),
                                          total=len(pcd_files)):
        pc, colors = read_point_cloud_and_labels(
            pcd_file, label_file, color_map)

        # Update AverageMeter
        if show_data_details:
            meters.update(pc)

        # Save as csv file (used by cloudcompare)
        save_array = np.concatenate((pc[:, :3], colors), axis=1)
        np.savetxt(pcd_file.replace("velodyne", "preds")[:-3] + "xyz",
                   save_array,
                   fmt="%f %f %f %d %d %d")

        # Project to range and save images
        range_res = proj({"Point": pc})
        range_img = range_res["Range"]["range_image"][..., 0]
        range_img = range_img / range_img.max() * 255
        img = colors[range_res["Range"]["p2r"]]
        cv2.imwrite(ROOT_PATH+f"images/image_{i:04d}.png", img)
        cv2.imwrite(ROOT_PATH+f"depth/image_{i:04d}.png", range_img)
    if show_data_details:
        print(meters)
    print("Plot images completely (if failed, please check permission)")


def videoPin(fourcc, scale, fps):
    # Sorted list of image files
    image_files = sorted(glob.glob(ROOT_PATH+'images/*.png'))

    # Get the frame size from the first image
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # Define the codec using VideoWriter_fourcc() and create a VideoWriter object

    if isinstance(fourcc, str):
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
    else:
        raise NotImplementError("args: fourcc should be a string with 4 char")
    video = cv2.VideoWriter(ROOT_PATH+'video.avi', fourcc,
                            fps, (width*scale[0], height*scale[1]))
    mix_video = cv2.VideoWriter(ROOT_PATH+'mix_video.avi', fourcc, 
                                fps, (width*scale[0], height*scale[1]*2))

    for image in tqdm(image_files):
        img = cv2.imread(image)
        depth_img = cv2.imread(image.replace("images", "depth"))
        mix_img = np.concatenate((img, depth_img), axis=0)
        mix_img = cv2.resize(mix_img, (width*scale[0], height*scale[1]*2))
        video.write(img)
        mix_video.write(mix_img)

    # Release the VideoWriter
    video.release()
    mix_video.release()
    print("Generate video completely (if failed, please check permission)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="visualize")
    parser.add_argument("-i", "--image", action="store_true", help="if visualize the predictions label in range images")
    parser.add_argument("-v", "--video", action="store_true", help="if pin images to video")
    parser.add_argument("-d", "--detail", action="store_ture", help="if show point clouds AverageMeter")
    args = parser.parse_args()
    if args.image:
        plotRangeImages(PROJ_HW[0], PROJ_HW[1], FOV_UP, FOV_DOWN, args.detail)
    if args.video:
        videoPin(FOURCC, SCALE, FPS)
    if not args.video and not args.image:
        print("Script need args such as: python plot.py -i -v")

