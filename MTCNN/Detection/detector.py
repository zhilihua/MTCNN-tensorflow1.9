import tensorflow as tf
import numpy as np

class Detector():
    #定义rnet或者onet，它们输入图片的尺寸为24或48(data_size)
    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph = tf.Graph()   #构造图
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input_image')
            self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
            self.sess = tf.Session()
            saver = tf.train.Saver()   #定义保存
            #加载模型参数
            saver.restore(self.sess, model_path)

        self.data_size = data_size
        self.batch_size = batch_size

    #rnet和onet测试
    def predict(self, databatch):
        #databatch：N*3*data_size*data_size
        scores = []
        batch_size = self.batch_size

        minibatch = []
        cur = 0
        n = databatch.shape[0]  #获取数据总数
        while cur < n:
            #将整体分为小批量数据
            minibatch.append(databatch[cur:min(cur+batch_size, n), :, :, :])
            cur += batch_size
        #每一批量的预测结果
        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            #最后一批处理情况,对数据进行循环扩展
            if m < batch_size:
                keep_inds = np.arange(m)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            cls_prob, bbox_pred, landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred, self.landmark_pred],
                                                           feed_dict={self.image_op: data})
            # 尺寸[num_batch * batch_size *2]
            cls_prob_list.append(cls_prob[:real_size])
            # 尺寸[num_batch * batch_size *4]
            bbox_pred_list.append(bbox_pred[:real_size])
            # 尺寸[num_batch * batch_size*10]
            landmark_pred_list.append(landmark_pred[:real_size])
            # 返回尺寸[num_of_data*2,num_of_data*4,num_of_data*10]

        return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(
        landmark_pred_list, axis=0)
