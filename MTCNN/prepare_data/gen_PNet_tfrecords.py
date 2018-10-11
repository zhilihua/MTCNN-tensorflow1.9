#生成pnet的tfrecords
import os
import random
import sys
import time

import tensorflow as tf

from tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple

def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    #从图片和注释中加载数据，并添加他们到tfrecord
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())

#获取数据
def get_dataset(dir, net='PNet'):
    item = 'imglists/PNet/train_%s_landmark.txt' % net

    dataset_dir = os.path.join(dir, item)
    imagelist = open(dataset_dir, 'r')

    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])

        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset

#获取输出文件的名称
def _get_output_filename(output_dir, name, net):
    return '%s/train_PNet_landmark.tfrecord' % (output_dir)

#运行操作
def run(dataset_dir, net, output_dir, name='MTCNN', shuffling=False):
    # 获取输出文件名字
    tf_filename = _get_output_filename(output_dir, name, net)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # 得到数据集并进行乱序
    dataset = get_dataset(dataset_dir, net=net)
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        random.shuffle(dataset)
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if i % 10000 == 0:
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(dataset)))
                sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)

    print('\nFinished converting the MTCNN dataset!')

if __name__ == '__main__':
    dir = '.'
    net = 'PNet'
    output_directory = 'imglists/PNet'
    run(dir, net, output_directory, shuffling=True)
