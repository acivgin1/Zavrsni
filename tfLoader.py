# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:40:17 2017

@author: Amar Civgin
"""
import tensorflow as tf
import matplotlib.pyplot as plt

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


def tf_loader(n_classes, batch_size, hm_epochs):
    # Getting the number of samples
    with open('filename lists/train1.txt', 'r') as train:
        train_lines = train.read().splitlines()

    # Reads pfathes of images together with their labels
    image_list, label_list = read_labeled_image_list('filename lists/train.txt')
    t_image_list, t_label_list = read_labeled_image_list('filename lists/test.txt')

    with tf.name_scope('loading'):
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        # t_images = tf.convert_to_tensor(t_image_list, dtype=tf.string)

        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        # t_labels = tf.convert_to_tensor(t_label_list, dtype=tf.int32)

        # Creating an input queue
        input_queue = tf.train.slice_input_producer([images, labels],
                                                    num_epochs=hm_epochs,
                                                    shuffle=False)
        # t_input_queue = tf.train.slice_input_producer([t_images, t_labels],
        #                                               num_epochs=hm_epochs,
        #                                               shuffle=False)

        image, label = read_images_from_disk(input_queue)
        # t_image, t_label = read_images_from_disk(t_input_queue)

        label = tf.one_hot(label, n_classes)
        # t_label = tf.one_hot(t_label, n_classes)

        image = tf.cast(image, tf.float32)
        image = tf.image.central_crop(image, .8125)
        image = tf.image.resize_images(image, [52, 52])

        # t_image = tf.cast(t_image, tf.float32)
        # t_image = tf.image.central_crop(t_image, .8125)
        # t_image = tf.image.resize_images(t_image, [52, 52])

        min_after_dequeue = batch_size
        capacity = min_after_dequeue + 3 * batch_size

        image, label = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)

        # t_image, t_label = tf.train.batch([t_image, t_label], batch_size=batch_size, capacity=capacity)

        return image, label, 0, 0, train_lines


def read_and_decode_single_example(filename, hm_epochs):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=hm_epochs,)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    # now return the converted data
    label = features['label']
    image_raw = features['image_raw']
    image = tf.decode_raw(image_raw, tf.float32)
    image = tf.reshape(image, (52, 52))

    # print(image.shape)

    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()
    #     tf.local_variables_initializer().run()  # loads images into x and y variables
    #     coord = tf.train.Coordinator()  # coordinator used for coordinating between various threads that load data
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # bitno
    #     image_raw = sess.run(image)
    #     plt.imshow(image_raw.squeeze(), cmap='gray')
    #     plt.show()
    #     coord.request_stop()
    #     coord.join(threads)
    return label, image


def tfrecords_loader(n_classes, batch_size, hm_epochs, train=True):
    if train:
        with tf.name_scope('tfRecordsloading'):
            label, image = read_and_decode_single_example("mnist.tfrecords", hm_epochs)

            label = tf.one_hot(label, n_classes)

            images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=20000,
                                                                min_after_dequeue=10000)
            return images_batch, labels_batch, 731668
    else:
        label, image = read_and_decode_single_example("mnist_test.tfrecords", hm_epochs)
        label = tf.one_hot(label, n_classes)
        images_batch, labels_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=82587,
                                                    allow_smaller_final_batch=True)
        return images_batch, labels_batch, 82587
if __name__ == '__main__':
    print('hi')
    tfrecords_loader(47, 128, 10)

print('Successfully imported tfLoader.')