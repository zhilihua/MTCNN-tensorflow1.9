import tensorflow as tf
import os
import cv2
from PIL import Image

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

#转换数据
def _conert_to_example(image_example, image_buffer, colorspace = b'RGB', channels=3, image_format=b'JPEG'):
    """
    :param image_example: 字典，一个图片实例
    :param image_buffer: 图片的编码格式
    :param colorspace:
    :param channels:
    :param image_format:
    :return:
    """
    class_label = image_example['label']
    image_bboxes = image_example.get('bbox', {})
    xmin = image_bboxes.get('xmin', [])
    xmax = image_bboxes.get('xmax', [])
    ymin = image_bboxes.get('ymin', [])
    ymax = image_bboxes.get('ymax', [])

    example = tf.train.Example(features=tf.train.Feature(feature={
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/format': _bytes_feature(image_format),
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/image_bbox/xmin': _float_feature(xmin),
        'image/image_bbox/ymin': _float_feature(ymin),
        'image/image_bbox/xmax': _float_feature(xmax),
        'image/image_bbox/ymax': _float_feature(ymax),
    }))
    return example

#另一种数据转换格式
def _convert_to_example_simple(image_example, image_buffer):
    class_label = image_example['label']
    bbox = image_example['bbox']
    roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    landmark = [bbox['xlefteye'], bbox['ylefteye'], bbox['xrighteye'], bbox['yrighteye'], bbox['xnose'], bbox['ynose'],
                bbox['xleftmouth'], bbox['yleftmouth'], bbox['xrightmouth'], bbox['yrightmouth']]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        'image/landmark': _float_feature(landmark)
    }))
    return example

class ImageCoder():
    #帮助类：提供tensorflow图片编码工具
    def __init__(self):
        self._sess = tf.Session()
        #初始化函数：转换png为jpeg
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        #初始化函数：编码RGB和JPEG数据
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    #转化图片数据从png到jpg
    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        # 编码图片数据为jpeg图片
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        return image

def _is_png(filename):
    #判断一个文件中是否包含png格式的文件
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() == '.png'

def _process_image(filename, coder):
    """处理单张图片文件
    Args:
      filename: string, 图片文件的路径，例如：'/path/to/example.JPG'.
      coder: 编码器
    Returns:
      image_buffer: string, 使用jpeg格式编码RGB图片
      height:
      width:
    """
    filename = filename + '.jpg'
    image = cv2.imread(filename)
    image_data = image.tostring()

    if _is_png(filename):
        image_data = coder.png_to_jpeg(image_data)
    height = image.shape[0]
    width = image.shape[1]
    return image_data, height, width

#处理无编码器的图片
def _process_image_withoutcoder(filename):
    image = cv2.imread(filename)
    image_data = image.tostring()

    height = image.shape[0]
    width = image.shape[1]

    return image_data, height, width