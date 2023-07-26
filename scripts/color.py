import yaml
import cv2
import numpy as np
import logging


logger = logging.getLogger("color_example")


def read_label_colors(file_path, swap_bgr=True):
    with open(file_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    color_map = data_loaded['color_map']

    # get label name list
    label_dict = data_loaded['labels']

    # filter valid labels
    valid_labels = list(data_loaded['learning_map_inv'].values())
    for label_id in list(color_map.keys()):
        if label_id not in valid_labels:
            color_map.pop(label_id)
            label_dict.pop(label_id)
    logger.debug(f"filted colormap: {color_map}")

    # swap colors from BGR to RGB

    if swap_bgr:
        for key in color_map:
            color_map[key] = color_map[key][::-1]

    return color_map, label_dict


def plot_color_example(yaml_path, root_path):
    # Load label colors
    color_map, label_dict = read_label_colors(yaml_path)
    color_example = np.zeros((200, 1000, 3), dtype=np.int32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # white color
    thickness = 1

    for i, (label_id, color) in enumerate(color_map.items()):
        row = i // 10
        column = i % 10
        color_example[row * 100: row * 100 + 100, 
                      column * 100: column * 100 + 100] = color
        label = label_dict[label_id]
        # adjust this to change the text position
        text_position = (column * 100 + 5, row * 100 + 50)
        cv2.putText(color_example, label, text_position, font, font_scale,
                    font_color, thickness, cv2.LINE_AA)

    cv2.imwrite(root_path + "color_example.png", color_example)


if __name__ == "__main__":
    plot_color_example("/home/cls2021/cvkdnk/workspace/PolarFusion/config/semantic-kitti.yaml", "./")

