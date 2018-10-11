import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
num_keep_radio = 0.7   #返现传播比例
#定义新的激励函数prelu
def prelu(inputs):
    '''
            xi   if xi > 0
    prelu =
            ai*xi  if xi <= 0
    '''
    alphas = tf.get_variable(name="alphas", shape=inputs.get_shape()[-1],
                             dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5   #若输入为正则该项为0，若为负则该项为本身
    return pos+neg

#将标签转化为one_hot类型的
def dense_to_one_hot(labels_dense, num_classes):
    num_labesls = labels_dense.shape[0]
    index_offset = np.arange(num_labesls)*num_classes  #获取每一个样本的起始位置
    labels_one_hot = np.zeros((num_labesls, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1   #将结果转换为一维的数组，使用时需要resize
    return labels_one_hot

#人脸非人脸分类损失函数，分类loss，加入困难数据挖掘，只关注正样本
def cls_ohem(cls_prob, label):
    #label的值为1或-1需要转换为0或1
    zeros = tf.zeros_like(label)
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    num_cls_prob = tf.size(cls_prob)  #获取预测结果总数，正样本可能和负样本可能之和
    cls_prob_reshape = tf.reshape(cls_prob, [num_cls_prob, -1])  #将预测值展开为一列
    label_int = tf.cast(label_filter_invalid, tf.int32)  #获取正样本的偏置
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*2
    indices_ = row + label_int   #获取每一个真实样本的索引
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))  #获取所有正样本的预测概率
    loss = -tf.log(label_prob + 1e-10)   #加偏置
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    valid_inds = tf.where(label < zeros, zeros, ones)
    num_valid = tf.reduce_sum(valid_inds)   #获取正样本的总数
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    # set 0 to invalid sample
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)   #选取前0.7的进行训练
    return tf.reduce_mean(loss)

#标签为-1或1需要规划，回归狂loss
def bbox_ohem(bbox_pred,bbox_target,label):
    #使用范数
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)

    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error,axis=1)

    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(num_valid, dtype=tf.int32)

    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

#面部轮廓loss值
def landmark_ohem(landmark_pred,landmark_target,label):
    #1代表正样本，0代表中间样本，-1代表负样本，-2代表面部轮廓值
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
#计算精度
def cal_accuracy(cls_prob, label):
    pred = tf.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int64)
    cond = tf.where(tf.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int, picked)
    pred_picked = tf.gather(pred, picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked, pred_picked), tf.float32))
    return accuracy_op

#定义P网络
def P_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        net = slim.conv2d(inputs, 10, 3, stride=1, scope='conv1')
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1', padding='SAME')
        net = slim.conv2d(net, num_outputs=16, kernel_size=[3, 3], stride=1, scope='conv2')
        net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=1, scope='conv3')
        # 尺寸为[batch*H*W*2]，预测是否为人脸
        conv4_1 = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1], stride=1, scope='conv4_1',
                              activation_fn=tf.nn.softmax)

        # 尺寸为[batch*H*W*4]，预测包围框的值
        bbox_pred = slim.conv2d(net, num_outputs=4, kernel_size=[1, 1], stride=1, scope='conv4_2', activation_fn=None)
        # 尺寸为[batch*H*W*10]，预测人脸坐标
        landmark_pred = slim.conv2d(net, num_outputs=10, kernel_size=[1, 1], stride=1, scope='conv4_3',
                                    activation_fn=None)
        if training:
            # 尺寸[batch*2]
            cls_prob = tf.squeeze(conv4_1, [1, 2], name='cls_prob')
            cls_loss = cls_ohem(cls_prob, label)
            # 尺寸[batch]
            bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
            # 尺寸[batch*10]
            landmark_pred = tf.squeeze(landmark_pred, [1, 2], name="landmark_pred")
            landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)

            accuracy = cal_accuracy(cls_prob, label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
            # test
        else:
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred, axis=0)
            landmark_pred_test = tf.squeeze(landmark_pred, axis=0)
            return cls_pro_test, bbox_pred_test, landmark_pred_test

#定义R网络
def R_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3, 3], stride=1, scope="conv1")
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=1, scope="conv2")
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        net = slim.conv2d(net, num_outputs=64, kernel_size=[2, 2], stride=1, scope="conv3")
        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128, scope="fc1", activation_fn=prelu)

        cls_prob = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc", activation_fn=tf.nn.softmax)
        bbox_pred = slim.fully_connected(fc1, num_outputs=4, scope="bbox_fc", activation_fn=None)
        landmark_pred = slim.fully_connected(fc1, num_outputs=10, scope="landmark_fc", activation_fn=None)

        if training:
            cls_loss = cls_ohem(cls_prob, label)
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
            accuracy = cal_accuracy(cls_prob, label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
        else:
            return cls_prob, bbox_pred, landmark_pred


def O_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3, 3], stride=1, scope="conv1")
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv2")
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv3")
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        net = slim.conv2d(net, num_outputs=128, kernel_size=[2, 2], stride=1, scope="conv4")
        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256, scope="fc1", activation_fn=prelu)
        cls_prob = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc", activation_fn=tf.nn.softmax)
        bbox_pred = slim.fully_connected(fc1, num_outputs=4, scope="bbox_fc", activation_fn=None)
        landmark_pred = slim.fully_connected(fc1, num_outputs=10, scope="landmark_fc", activation_fn=None)

        if training:
            cls_loss = cls_ohem(cls_prob, label)
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
            accuracy = cal_accuracy(cls_prob, label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
        else:
            return cls_prob, bbox_pred, landmark_pred
