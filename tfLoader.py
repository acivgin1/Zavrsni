# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:40:17 2017

@author: Amar Civgin
"""
import io
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from zipfile import ZipFile

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
        filename_and_label_tensor: A scalar string tensor.
    Returns:
        Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=1)
    return example, label


def zip_loader(filename, n_classes, batch_size, load_train=True, shuffle=True, current=0):
    with ZipFile(filename) as archive:
        if load_train:
            with open('filename lists/train2.txt') as train:
                zip_loader.train_list = train.readlines()

                if shuffle:
                    random.shuffle(zip_loader.train_list)
                images = None
                labels = []
                i = 0
                for line in zip_loader.train_list[current: current + batch_size]:
                    filename, label = line[:-1].split(' ')

                    # loading image and converting from bytes to image to np array
                    byte_content = archive.read(filename[3:].rstrip())

                    image_content = Image.open(io.BytesIO(byte_content))
                    image_content = image_content.convert('L')
                    image_content = image_content.crop((12, 12, 116, 116))
                    image_content = image_content.resize((52, 52), Image.BILINEAR)

                    arr = np.asarray(image_content, dtype='float32')
                    arr = np.reshape(arr, (1, 52, 52, 1))
                    i = i+1
                    if images is None:
                        images = arr
                    else:
                        images = np.concatenate((images, arr), axis=0)
                    labels.append(int(label))

                # converting labels from int to one hot vector
                labels_int = np.array(labels)
                n_labels = labels_int.shape[0]
                index_offset = np.arange(n_labels) * n_classes
                labels = np.zeros((n_labels, n_classes))
                labels.flat[index_offset + labels_int.ravel()] = 1
                print(i)
                return images, labels, zip_loader.train_list

        else:
            with open('filename lists/test2.txt') as test:
                test_list = test.readlines()
                t_images = None
                t_labels = []
                i = 0
                for line in test_list:
                    filename, t_label = line[:-1].split(' ')
                    # loading image and converting from bytes to image to np array
                    byte_content = archive.read(filename[3:].rstrip())
                    image_content = Image.open(io.BytesIO(byte_content))
                    image_content = image_content.convert('L')
                    arr = np.asarray(image_content, dtype='float32')
                    arr = np.reshape(arr, (1, 128, 128, 1))

                    i = i + 1
                    if t_images is None:
                        t_images = arr
                    else:
                        t_images = np.concatenate((t_images, arr), axis=0)
                    if i > 256:
                        break

                    t_labels.append(int(t_label))

                # converting labels from int to one hot vector
                t_labels_int = np.array(t_labels)
                n_t_labels = t_labels_int.shape[0]
                index_offset = np.arange(n_t_labels) * n_classes
                t_labels = np.zeros((n_t_labels, n_classes))
                t_labels.flat[index_offset + t_labels_int.ravel()] = 1
                return t_images, t_labels


def tf_loader(n_classes, batch_size, hm_epochs):
    # Getting the number of samples
    with open('filename lists/train2.txt', 'r') as train:
        train_lines = train.read().splitlines()

    # Reads pfathes of images together with their labels
    image_list, label_list = read_labeled_image_list('filename lists/train2.txt')
    t_image_list, t_label_list = read_labeled_image_list('filename lists/test2.txt')

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    t_images = tf.convert_to_tensor(t_image_list, dtype=tf.string)

    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    t_labels = tf.convert_to_tensor(t_label_list, dtype=tf.int32)

    # Creating an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=hm_epochs,
                                                shuffle=True)
    t_input_queue = tf.train.slice_input_producer([t_images, t_labels],
                                                shuffle=True)

    image, label = read_images_from_disk(input_queue)
    t_image, t_label = read_images_from_disk(t_input_queue)

    label = tf.one_hot(label, n_classes)
    t_label = tf.one_hot(t_label, n_classes)

    image = tf.cast(image, tf.float32)
    image = tf.image.central_crop(image, .8125)
    image = tf.image.resize_images(image, [52, 52])

    # with tf.Session() as sess:
    #     sess.run(tf.local_variables_initializer())  #bitno
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess = sess, coord = coord)          #bitno
    #     image, label = sess.run([image, label])
    #
    #     plt.imshow(image.squeeze(), cmap='gray')
    #     plt.show()
    #     print(image.shape)
    #
    #
    #     coord.request_stop()
    #     coord.join(threads)
    # sys.exit("Error message")

    t_image = tf.cast(t_image, tf.float32)
    t_image = tf.image.central_crop(t_image, .8125)
    t_image = tf.image.resize_images(t_image, [52, 52])
    
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    image, label = tf.train.shuffle_batch([image, label], batch_size=2048, capacity=capacity,
                                          min_after_dequeue=min_after_dequeue)

    t_image, t_label = tf.train.shuffle_batch([t_image, t_label], batch_size=2048, capacity=capacity,
                                          min_after_dequeue=min_after_dequeue)

    return image, label, t_image, t_label, train_lines
print('Successfully imported tfLoader.')

if __name__ == '__main__':
    print('Executing code inside tfLoader main')
    zip_loader('D:/by_merge.zip', 47, 10)