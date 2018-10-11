import os
from os.path import join, exists
import time
import cv2
import numpy as np

def createDir(p):
    if not os.path.exists(p):
        os.mkdir(p)
#用相同的方式乱序
def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

#在图中做标记
def drawLandmark(img, bbox, landmark):
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 0, 255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 2), -1)
    return img

#从txt文件获取数据
def getDataFromTxt(txt, with_landmark=True):
    """
    :param txt:
    :param with_landmark:
    :return: [(img_path, bbox, landmark)]
    其中bbox:[left, right, top, bottom]
    landmark:[(x1, y1),(x2, y2),...]
    """
    dirname = os.path.dirname(txt)
    with open(txt, 'r') as fd:
        lines = fd.readlines()
    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(dirname, components[0])   #获取文件路径
        bbox = (components[1], components[3], components[2], components[4])
        bbox = [float(_) for _ in bbox]   #将数据转换为float类型
        bbox = list(map(int, bbox))
        #坐标
        if not with_landmark:
            result.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))
        for index in range(5):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv
        result.append((img_path, BBox(bbox), landmark))
    return result

def getPatch(img, bbox, point, padding):
    point_x = bbox.x + point[0] * bbox.w
    point_y = bbox.y + point[1] * bbox.h
    patch_left = point_x - bbox.w * padding
    patch_right = point_x + bbox.w * padding
    patch_top = point_y - bbox.h * padding
    patch_bottom = point_y + bbox.h * padding
    patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
    patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
    return patch, patch_bbox

#预处理图片
def processImage(imgs):
    #imgs:[N,1,W,H]
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        imgs[i] = (img - 127.5) / 128
    return imgs

#定义Bbox类
class BBox():
    def __init__(self, bbox):
        bbox = list(map(int, bbox))
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]

        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]

    #对包围框进行扩展
    def expand(self, scale=0.05):
        bbox = [self.left, self.top, self.right, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] -= int(self.h * scale)
        bbox[2] += int(self.w * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    #点的偏置[0, 1]
    def project(self, point):
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    #绝对位置
    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])
    #坐标
    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p
    # 根据bbox改变偏置
    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, top, right, bottom])
