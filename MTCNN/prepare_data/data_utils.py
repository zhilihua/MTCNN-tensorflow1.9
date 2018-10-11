import os
import numpy as np
import cv2

#读取注释并提取相关信息
def read_annotation(base_dir, label_path):
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        images.append(imagepath)
        nums = labelfile.readline().strip('\n')    #获取包围框的个数
        one_image_bboxes = []   #存放每一张图片的包围框
        for i in range(int(nums)):
            bb_info = labelfile.readline().strip('\n').split(' ')
            face_box = [float(bb_info[i]) for i in range(4)]
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
        bboxes.append(one_image_bboxes)
    data['images'] = images  #保存所有的图片路径
    data['bboxes'] = bboxes  #保存所有包围框
    return data

def get_path(base_dir, filename):
    return os.path.join(base_dir, filename)
