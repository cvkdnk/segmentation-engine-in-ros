import numpy as np
import os
import yaml
import cv2

import utils



data_path = "/root/autodl-tmp/dataset/sequences/00/velodyne/"
results_path = "/root/autodl-tmp/results/0217/pred/"
kitti_yaml_path = "/root/autodl-nas/polar_fusion/config/semantic-kitti.yaml"

kitti_yaml = yaml.safe_load(open(kitti_yaml_path, 'r'))
rp = utils.RangeProject()


for filename in os.listdir(results_path):
    frame = filename[3:9]
    filepath = os.path.join(results_path, filename)
    pred = np.loadtxt(filepath).astype(np.int32)
    mapping = np.loadtxt(filepath.replace("pred", "map")).astype(np.int32)
    keep_index = np.loadtxt(filepath.replace("pred", "keep")).astype(np.int32)
    binfile = data_path+frame+".bin"
    points, sem_labels, *_ = utils.load_data(binfile)
    points = points[:, :3]

    # 降采样后的预测标签投影回原始场景中所有点的预测标签
    print("mapping size {}, max {}, min {}".format(mapping.size, mapping.max(), mapping.min()))
    print("pred size {}, max {}, min {}".format(pred.size, pred.max(), pred.min()))
    map_pred = pred[mapping]
    map_pred = utils.label_mapping(map_pred, kitti_yaml["learning_map_inv"])
    assert(sem_labels.size == map_pred.size)

    acc = len((sem_labels == map_pred).nonzero()) / (sem_labels.size+0.01)
    data = {"Point": points}
    range_dict = rp(data)
    p2r = range_dict["Range"]["p2r"]
    print("p2r.shape: ", p2r.shape)

    map_dict = kitti_yaml["color_map"]
    for i in map_dict:
        map_dict[i] = np.array(map_dict[i], dtype=np.int32)
    
    raw_pc = utils.color_map(sem_labels, map_dict)
    pred_pc = utils.color_map(map_pred, map_dict)

    raw_range_img = raw_pc[p2r]
    pred_range_img = pred_pc[p2r]
    print("raw img shape: ", raw_range_img.shape)

    save_path = filepath.replace("pred", "img")
    save_path = save_path + filename.replace("txt", "jpg")

    save_img = np.concatenate((raw_range_img, pred_range_img), axis=0)
    # cv2.putText(save_img, "{:.2f}%".format(acc*100), (0, 100), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

    cv2.imwrite(save_path, save_img)





