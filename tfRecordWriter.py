# load up some dataset. Could be anything but skdata is convenient.
import random
import io

import numpy as np
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


with ZipFile('D:/by_merge.zip') as archive:
    with open('filename lists/train1.txt') as train:
        train_list = train.readlines()
        random.shuffle(train_list)

        writer = tf.python_io.TFRecordWriter('images.tfrecord')
        for line in tqdm(train_list[:]):
            filename, label = line[:-1].split(' ')

            # loading image and converting from bytes to image to np array
            byte_content = archive.read(filename[3:].rstrip())

            image_content = Image.open(io.BytesIO(byte_content))
            image_content = image_content.convert('L')
            image_content = image_content.crop((12, 12, 116, 116))
            image_content = image_content.resize((52, 52), Image.BILINEAR)

            arr = np.asarray(image_content, dtype='float32')
            arr = np.reshape(arr, (1, 52, 52, 1))

            shape = np.array(arr.shape, np.int32)
            shape = shape.tostring()

            arr = arr.tostring()

            # converting labels from int to one hot vector
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(int(label)),
                'shape': _bytes_feature(shape),
                'image': _bytes_feature(arr)
            }))
            writer.write(example.SerializeToString())
        writer.close()