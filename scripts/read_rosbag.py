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


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

bag = rosbag.Bag('/root/autodl-tmp/f1_2023-07-05-17-01-17.bag')

rosbag_info = bag.get_type_and_topic_info()

# Add an empty line after import statements
for i, (topic, topic_info) in enumerate(rosbag_info.topics.items()):
   # Add a number for each topic
   logging.info(f"{i+1}.")
   # Use f-string to format strings
   logging.info(f"Topic: {topic}")
   logging.info(f"Type: {topic_info.msg_type}")
   logging.info(f"Message count: {topic_info.message_count}")
   logging.info(f"Connections: {topic_info.connections}")
   # Use round function to keep two decimal places
   logging.info(f"Frequency: {round(topic_info.frequency, 2)}")
   logging.info("")

for i, (topic, msg, t) in tqdm(enumerate(bag.read_messages(topics=["/velodyne_points"]))):
    # logging.debug(f"msg type {type(msg)}")
    cloud_points = pc2.read_points(msg, field_names=["x","y","z", "intensity"], skip_nans=True)
    points_array = np.array(list(cloud_points))
    points_array = points_array.astype(np.float32)
    # logging.debug(f"points_array.dtype: {points_array.dtype}")
    assert points_array.dtype == np.float32
    points_array.tofile(f"/root/autodl-tmp/data/{i}_bin")


bag.close()

