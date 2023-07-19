import numpy as np
import open3d as o3d
import glob
import cv2
import yaml
import os
from tqdm import tqdm

from proj import RangeProject, label_mapping

def read_label_colors(file_path):
    with open(file_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    color_map = data_loaded['color_map']
    # swap colors from BGR to RGB
    for key in color_map:
        color_map[key] = color_map[key][::-1]
    return color_map


def cart2spherical(input_xyz):
    """return rho, phi, theta; also known as depth, yaw, pitch"""
    depth = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2 + input_xyz[..., 2] ** 2)
    yaw = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    pitch = np.arcsin(input_xyz[..., 2] / depth)
    return np.stack((depth, yaw, pitch), axis=-1)


def read_point_cloud_and_labels(pcd_path, label_path, color_map):
    point_cloud = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    sph_coords = cart2spherical(point_cloud)
    labels = np.fromfile(label_path, dtype=np.int32)
    with open("semantic-kitti.yaml", 'r') as f:
        kitti_config = yaml.safe_load(f)
    # Create an empty list to store RGB colors
    colors = np.empty([len(labels), 3])

    # Color the point cloud based on the labels
    for label_id, color in color_map.items():
        colors[labels == label_id] = color

    return point_cloud, colors

# Load label colors
color_map = read_label_colors('semantic-kitti.yaml')

# Directory paths
pcd_dir = './velodyne/'
label_dir = './predictions/'

# Sorted list of point cloud and label files
pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.bin')))
label_files = sorted(glob.glob(os.path.join(label_dir, '*.label')))

# Store point clouds
point_clouds = []

# Create a directory to store images
os.makedirs('images', exist_ok=True)

# Init RangeProject
proj = RangeProject(proj_H=16, proj_W=512, proj_fov_up=15, proj_fov_down=-15)

for i, (pcd_file, label_file) in tqdm(enumerate(zip(pcd_files, label_files)), total=len(pcd_files)):
    pc, colors = read_point_cloud_and_labels(pcd_file, label_file, color_map)
    print("max", pc.max(axis=0))
    print("min", pc.min(axis=0))
    print("mean", pc.mean(axis=0))
    print("std", pc.std(axis=0))

    save_array =np.concatenate((pc[:, :3], colors), axis=1)
    np.savetxt(pcd_file.replace("velodyne", "preds")[:-3]+"xyz", save_array, fmt="%f %f %f %d %d %d")
    range_res = proj({"Point": pc})
    range_img = range_res["Range"]["range_image"][..., 0]
    print("mean: ", np.mean(range_res["Range"]["range_image"], axis=(0, 1)))
    print("std: ", range_res["Range"]["range_image"].std(axis=(0, 1)))
    range_img = range_img / range_img.max() * 255
    img = colors[range_res["Range"]["p2r"]]
    cv2.imwrite(f"images/image_{i:04d}.png", img)
    cv2.imwrite(f"depth/image_{i:04d}.png", range_img)

# Sorted list of image files
image_files = sorted(glob.glob('images/*.png'))

# Get the frame size from the first image
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

# Define the codec using VideoWriter_fourcc() and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter('video.avi', fourcc, 5.0, (width, height))
mix_video = cv2.VideoWriter('mix_video.avi', fourcc, 5.0, (width, height*4))


for image in image_files:
    img = cv2.imread(image)
    depth_img = cv2.imread(image.replace("images", "depth"))
    mix_img = np.concatenate((img, depth_img), axis=0)
    mix_img = cv2.resize(mix_img, (width, height*4))
    video.write(img)
    mix_video.write(mix_img)


# Release the VideoWriter
video.release()

