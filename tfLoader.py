# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:40:17 2017

@author: Amar Civgin
"""
import tensorflow as tf

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
        filename, label = line[:-1].split('#')
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

def loader(n_classes, batch_size, hm_epochs):
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

    '''with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())  #bitno
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)          #bitno
        image, label = sess.run([image, label])
        
        plt.imshow(image.squeeze(), cmap='gray')
        plt.show()
        print(image.shape)
        
        
        coord.request_stop()
        coord.join(threads)
    sys.exit("Error message")
    '''
    
    t_image = tf.cast(t_image, tf.float32)
    t_image = tf.image.central_crop(t_image, .8125)
    t_image = tf.image.resize_images(t_image, [52, 52])
    
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    image, label = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                          min_after_dequeue=min_after_dequeue)

    t_image, t_label = tf.train.shuffle_batch([t_image, t_label], batch_size=batch_size, capacity=capacity,
                                          min_after_dequeue=min_after_dequeue)

    return image, label, t_image, t_label, train_lines
print('Successfully imported tfLoader.')