#将三个检测器汇集在一起
import cv2
import time
import numpy as np
import sys
sys.path.append("../")
from train_models.MTCNN_config import config
from .nms import py_nms

class MtcnnDetector:
    def __init__(self,
                 detectors,   #探测器模型存放器，P、R、O模型
                 min_face_size=25,   #能够检查的最小人脸的尺寸
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],   #
                 scale_factor=0.79,  #原图缩放比例
                 slide_window=False
                 ):
        self.pnet_detector = detectors[0]    #获得pnet
        self.rnet_detector = detectors[1]    #获得rnet
        self.onet_detector = detectors[2]    #获得onet
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
        self.slide_window = slide_window

    #转化包围框形状
    def convert_to_square(self, bbox):
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
            进行转化
        参数解释:
        ----------
            bbox: 形状[n x 5]
            reg:  形状[n x 4]
        返回:
        -------
            变化后的包围框
        """
        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c

    #生成包围框
    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
        从特征图中生成包围框
        :param cls_map: 形状[n x m]，每一个位置的探测得分
        :param reg: 形状[n x m x 4]
        :param scale:
        :param threshold:
        :return: 包围框
        """
        stride = 2  #在全网络过程中，有一个pool步长为2，在反推过程中应该乘以2
        cellsize = 12   #最小人脸的大小

        t_index = np.where(cls_map > threshold)   #获取符合的包围框索引
        if t_index[0].size == 0:
            return np.array([])
        #获取偏置
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]
        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        #获取包围框信息，由于读取时为（y,x）。本网络应用的是全连接网络，所以特征图的每一个点
        #可以作为包围框的起点
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])
        return boundingbox.T

    #图片预处理
    def processed_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
        img_resized = (img_resized - 127.5) / 128
        return img_resized

    def pad(self, bboxes, w, h):
        """
            填补bboxes，也限制它的尺寸
        参数:
        ----------
            bboxes: 尺寸[n x 5]
            w: 输入图片的宽度
            h: 输入图片的高度
        Returns :
        ------
            dy, dx : 尺寸[n x 1]，目标图片的起始点
            edy, edx : 尺寸[n x 1]，目标图片的终止点
            y, x : 尺寸[n x 1]，原始图片的起始点
            ey, ex : 尺寸[n x 1]，原始图片的终止点
            tmph, tmpw: 尺寸[n x 1]，边界框的高和宽
        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]
        # 由于在rnet和onet中输入图片为挑选后的包围框所以假设为整个包围框的大小
        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        #当遇到超越边界情况下做调整
        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """通过pnet得到人脸的候选框
        参数:
        ----------
        im: 输入图片
        返回:
        -------
        boxes: 标定前的探测框
        boxes_c: 标定后的探测框
        """
        h, w, c = im.shape
        net_size = 12

        current_scale = float(net_size) / self.min_face_size  # 设定识别的最小尺寸
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        all_boxes = list()
        while min(current_height, current_width) > net_size:
            # 通过预测返回结果
            # cls_cls_map : 尺寸：H*w*2
            # reg: 尺寸：H*w*4
            cls_cls_map, reg = self.pnet_detector.predict(im_resized)
            # boxes: 尺寸：num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
            boxes = self.generate_bbox(cls_cls_map[:, :, 1], reg, current_scale, self.thresh[0])

            current_scale *= self.scale_factor  #进行图片的金字塔缩放，产生多维度的图片
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None, None

        all_boxes = np.vstack(all_boxes)

        # 融合第一步产生的探测
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # 对边界框进行调整
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None

    def detect_rnet(self, im, dets):
        """通过rnet得到人脸候选框
        参数:
        ----------
        im: 输入图片
        dets: 从pnet得到的探测结果
        返回:
        -------
        boxes: 矫正前的探测框
        boxes_c: 矫正后的探测框
        """
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)   #获取矫正边界框
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128
        # cls_scores : 尺寸：num_data*2
        # reg: 尺寸：num_data*4
        # landmark: 尺寸：num_data*10
        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            # landmark = landmark[keep_inds]
        else:
            return None, None, None

        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep])  #进行边界框矫正
        return boxes, boxes_c, None

    def detect_onet(self, im, dets):
        """通过rnet得到人脸候选框
        参数:
        ----------
        im: 输入图片
        dets: 从pnet得到的探测结果
        返回:
        -------
        boxes: 矫正前的探测框
        boxes_c: 矫正后的探测框
        """
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

        cls_scores, reg, landmark = self.onet_detector.predict(cropped_ims)

        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[2])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None

        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        #通过坐标的比例位置得到相应包围框的位置
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)

        boxes = boxes[py_nms(boxes, 0.6, "Minimum")]
        keep = py_nms(boxes_c, 0.6, "Minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c, landmark
    #检测视频
    def detect(self, img):
        #在图片上探测人脸
        # pnet
        if self.pnet_detector:
            boxes, boxes_c, _ = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([]), np.array([])

        # rnet
        t2 = 0
        if self.rnet_detector:
            boxes, boxes_c, _ = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])

        # onet
        if self.onet_detector:
            boxes, boxes_c, landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])

        return boxes_c, landmark

    def detect_face(self, test_data):
        all_boxes = []  # 保存每一张图片的bboxes
        landmarks = []
        for databatch in test_data:
            im = databatch
            # pnet
            if self.pnet_detector:
                # 忽略landmark
                boxes, boxes_c, landmark = self.detect_pnet(im)
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    continue
            # rnet
            if self.rnet_detector:
                # 忽略landmark
                boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    continue
            # onet
            if self.onet_detector:
                boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    continue

            all_boxes.append(boxes_c)
            landmarks.append(landmark)

        return all_boxes, landmarks
