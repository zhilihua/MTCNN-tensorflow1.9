import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU
anno_file = "wider_face_train.txt"   #关于图片的注解文件
im_dir = "WIDER_train/images"     #图片所在路径
pos_save_dir = "12/positive"    #正样本存放位置
part_save_dir = "12/part"    #中间样本存放位置
neg_save_dir = "12/negative"   #负样本存放位置
save_dir = "./12"    #样本存在的文件夹
#判断文件是否存在，不存在则创建
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

#打开文件准备向其中写入数据
f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
p_idx = 0  #正样本索引
n_idx = 0  #负样本索引
d_idx = 0  #不关心样本索引
idx = 0
box_idx = 0
#剪切保存图片并注解
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    #获取图片路径
    im_path = annotation[0]
    #将所有的包围框值转换为float
    bbox = map(float, annotation[1:])
    #将包围框转换为矩阵
    boxes = np.array(list(bbox), dtype=np.float32).reshape(-1, 4)
    #加载图片
    img = cv2.imread(os.path.join(im_dir, im_path+'.jpg'))
    idx += 1
    height, width, channel = img.shape
    #生成负样本
    neg_num = 0
    while neg_num < 50:
        size = npr.randint(12, min(width, height)/2)
        #左上定点坐标
        nx = npr.randint(0, width-size)
        ny = npr.randint(0, height-size)
        #剪切图片
        crop_box = np.array([nx, ny, nx+size, ny+size])
        #计算iou
        Iou = IoU(crop_box, boxes)   #计算生成包围框与所有真实框的iou

        cropped_im = img[ny: ny+size, nx: nx+size, :]
        resize_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            save_dir = os.path.join(neg_save_dir, "%s.jpg" %n_idx)
            f2.write("12/negative/%s.jpg"%n_idx+' 0\n')  #文档中记录样本label
            cv2.imwrite(save_dir, resize_im)
            n_idx += 1
            neg_num += 1
    #再次寻找正负、和中间样本
    for box in boxes:
        #真实包围框的值
        x1, y1, x2, y2 = box
        #真实包围框的宽和高
        w = x2-x1+1
        h = y2-y1+1
        #忽略小于40的脸
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        for i in range(5):
            size = npr.randint(12, min(width, height)/2)
            #获取随机偏置
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1: ny1+size, nx1: nx1+size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write("12/negative/%s.jpg" % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
        #生成正样本和中间样本
        for i in range(20):
            #尺寸为[minsize*0.8, maxsize*1.25]
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)
            #nx1 = max(x1+w/2-size/2+delta_x)
            nx1 = int(max(x1 + w/2 + delta_x - size/2, 0))
            ny1 = int(max(y1 + h/2 + delta_y -size/2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                 continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            # 与真实框的偏置比例
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[ny1: ny2, nx1: nx2, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write("12/positive/%s.jpg" % p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (
                offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write("12/part/%s.jpg" % d_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (
                offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1

f1.close()
f2.close()
f3.close()
