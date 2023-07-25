import argparse
import rosbag
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import logging
import numpy as np
from tqdm import tqdm
'''
Topic: /velodyne_points
Type: sensor_msgs/PointCloud2
Message count: 2533
Connections: 1
Frequency: 9.91
'''

# e.g. python read_rosbag.py -s

# CHANGE IT
save_path = "/root/autodl-tmp/sequences/11/velodyne/"
bag_path = "/root/autodl-tmp/f1_2023-07-05-17-01-17.bag"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def showTopicInfo(bag):
    rosbag_info = bag.get_type_and_topic_info()

    for i, (topic, topic_info) in enumerate(rosbag_info.topics.items()):
        # Add a number for each topic
        logging.info(f"{i+1}.")
        # Use f-string to format strings
        logging.info(f"Topic: {topic}")
        logging.info(f"Type: {topic_info.msg_type}")
        logging.info(f"Message count: {topic_info.message_count}")
        logging.info(f"Connections: {topic_info.connections}")
        # Use round function to keep two decimal places
        logging.info(f"Frequency: {round(topic_info.frequency, 2)}\n")


def loadVelodynePoints(bag, save_path=None):
    for i, (topic, msg, t) in tqdm(
            enumerate(bag.read_messages(topics=["/velodyne_points"]))):
        # logging.debug(f"msg type {type(msg)}")
        cloud_points = pc2.read_points(
            msg, field_names=["x", "y", "z", "intensity"], skip_nans=True)
        points_array = np.array(list(cloud_points))
        points_array = points_array.astype(np.float32)
        # print("="*80)
        # print(f"intensity: {points_array.max(axis=0)} {points_array.min(axis=0)} {points_array.mean(axis=0)}")
        assert points_array.dtype == np.float32
        points_array[:, 3] /= 255.0

        if save_path is not None:
            points_array.tofile(save_path + f"{i}.bin")
        # yield points_array
        
def load_image(bag, save_path=None):
    bridge = CvBridge()
    for i, (topic, msg, t) in tqdm(
            enumerate(bag.read_messages('/camera/rgb/image_rect_color'))):
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(save_path.replace('velodyne','camera')+f"{i}.bin",cv_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='read_rosbag')
    parser.add_argument('-s',
                        '--show',
                        action='store_true',
                        help="show the ros topic information")
    parser.add_argument('-x',
                        '--xload',
                        action="store_true",
                        help="dont load and save the velodyne points")
    args = parser.parse_args()

    bag = rosbag.Bag(bag_path)

    if args.show:
        showTopicInfo(bag)

    if not args.xload:
        loadVelodynePoints(bag, save_path)
        load_image(bag, save_path)

    bag.close()
