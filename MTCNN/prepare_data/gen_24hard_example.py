import sys
sys.path.insert(0,'..')
import numpy as np
import argparse
import os
import pickle
import cv2
from train_models.mtcnn_model import P_Net,R_Net
from train_models.MTCNN_config import config
from prepare_data.loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from prepare_data.utils import *
from prepare_data.data_utils import *

def t_net(prefix, epoch, batch_size, test_mode='PNet', thresh=[0.6, 0.6, 0.6],
          min_face_size=25, stride=2, slide_window=False, shuffle=False, vis=False):
    """
    :param prefix: 模型名字的前缀
    :param epoch:上一个模型的训练次数
    :param batch_size:用在预测中的批的尺寸
    :param test_mode:测试模式
    :param thresh:阈值
    :param min_face_size:最小检测人脸大小
    :param stride:
    :param slide_window:是否在pnet中应用了sliding window
    :param shuffle:
    :param vis:
    :return:
    """
    detectors = [None, None, None]
    model_path = ['%s-%s' %(x, y) for x, y in zip(prefix, epoch)]
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    basedir = '.'
    # 注解文件
    filename = './wider_face_train_bbx_gt.txt'

    data = read_annotation(basedir, filename)
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    test_data = TestLoader(data['images'])
    detections, _ = mtcnn_detector.detect_face(test_data)

    save_net = 'RNet'
    if test_mode == "PNet":
        save_net = "RNet"
    elif test_mode == "RNet":
        save_net = "ONet"
    # 保存探测结果
    save_path = os.path.join(data_dir, save_net)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(detections, f, 1)
    print("%s测试完成开始OHEM" % image_size)
    save_hard_example(image_size, data, save_path)

def save_hard_example(net, data, save_path):
    im_idx_list = data['images']  #获取图片路径
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    #保存文件
    neg_label_file = "%d/neg_%d.txt" %(net, image_size)
    neg_file = open(neg_label_file, 'w')

    pos_label_file = "%d/pos_%d.txt" % (net, image_size)
    pos_file = open(pos_label_file, 'w')

    part_label_file = "%d/part_%d.txt" % (net, image_size)
    part_file = open(part_label_file, 'w')

    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb')) #加载PNet中的探测框
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        # 改变包围框的形状
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 忽略太小或者超过边界的包围框
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3 and neg_num < 60:
                save_file = get_path(neg_dir, "%s.jpg" % n_idx)
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                if np.max(Iou) >= 0.65:
                    save_file = get_path(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()

if __name__ == "__main__":
    image_size = 24

    base_dir = '../prepare_data/WIDER_train'
    data_dir = '%s' % str(image_size)   #数据的保存目录

    neg_dir = get_path(data_dir, 'negative')
    pos_dir = get_path(data_dir, 'positive')
    part_dir = get_path(data_dir, 'part')
    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    t_net(['../data/MTCNN_model/PNet_landmark/PNet',
           '../data/MTCNN_model/RNet_landmark/RNet',
           '../data/MTCNN_model/ONet/ONet'],  # 模型参数文件
          [30, 14, 22],  # 最总循环轮数（暂定）
          [2048, 256, 16],  # 测试的尺寸
          'PNet',  # 测试模式
          [0.4, 0.5, 0.7],  # 分类阈值
          24, 2,  False, False, vis=False)
