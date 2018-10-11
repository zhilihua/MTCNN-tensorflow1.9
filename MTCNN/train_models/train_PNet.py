from mtcnn_model import P_Net
from train import train


def train_PNet(base_dir, prefix, end_epoch, display, lr):
    """
    训练 PNet
    :param dataset_dir: tfrecord 路径
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = P_Net
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr)


if __name__ == '__main__':
    # 数据路径
    base_dir = '../prepare_data/imglists/PNet'
    model_name = 'MTCNN'
    model_path = '../data/%s_model/PNet_landmark/PNet' % model_name

    prefix = model_path
    end_epoch = 30
    display = 100
    lr = 0.01
    train_PNet(base_dir, prefix, end_epoch, display, lr)
