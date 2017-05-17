import tensorflow as tf
import datetime
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data", one_hot=True)



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


n_classes = 47
batch_size = 64

x = tf.placeholder(tf.float32, [None, 86*86])
y = tf.placeholder(tf.float32, [None, n_classes])

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal( [input_dim, output_dim] ))
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.random_normal( [output_dim] ))
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name = 'activations')
        tf.summary.histogram('activations', activations)
        return activations

def conv_layer(input_tensor, filter_height, filter_width, in_channels, out_channels, strides, padding, layer_name, act = tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('window'):
            conv_filter = tf.Variable(tf.random_normal([filter_height, filter_width, in_channels, out_channels]))
            variable_summaries(conv_filter)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.random_normal( [out_channels] ))
            variable_summaries(biases)
        with tf.name_scope('W_conv_x_plus_b'):
            convolution = tf.nn.conv2d(input_tensor, conv_filter, strides = strides, padding = padding) + biases
            tf.summary.histogram('convolution', convolution)
        activatedConv = act(convolution, name = 'activated_convolution')
        tf.summary.histogram('activated_convolution', activatedConv)
        return activatedConv

def max_pool_layer(input_tensor, ksize, strides, padding, layer_name):
    with tf.name_scope(layer_name):
        max_pool = tf.nn.max_pool(input_tensor, ksize = ksize, strides = strides, padding = padding, name = 'max_pooling')
        tf.summary.histogram('max_pool', max_pool)
        return max_pool

def convolutional_neural_network(data):
    data = tf.reshape(data, shape = [-1, 86, 86, 1], name = 'input' )

    #conv1 = conv_layer(data, filter_height = 5, filter_width = 5, in_channels = 1, out_channels = 8, strides = [1,1,1,1], padding = 'SAME', layer_name = 'conv1')
    conv1 = conv_layer(data, 5, 5, 1, 8, [1, 1, 1, 1], 'SAME', 'conv1')
    conv1 = max_pool_layer(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', layer_name = 'max_pool1')

    conv2 = conv_layer(conv1, 5, 5, 8, 32, [1, 1, 1, 1], 'SAME', 'conv2')
    conv2 = max_pool_layer(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', layer_name = 'max_pool2')

    #conv3 = conv_layer(conv2, 5, 5, 32, 16, [1,1,1,1], 'SAME', 'conv3')
    #conv3 = max_pool_layer(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', layer_name = 'max_pool3')
    
    shape_dim = 22*22*32
    fc1 = tf.reshape(conv2, shape = [-1, shape_dim], name = 'conv2_maxpool3' )
    fc1 = nn_layer(fc1, shape_dim, 128, 'fully_connected1')
    #fc1 = tf.nn.dropout(fc1, 0.85)

    #fc2 = nn_layer(fc1, 1024, 128, 'fully_connected2')

    with tf.name_scope('output'):
        with tf.name_scope('output_weights'):
            weights = tf.Variable(tf.random_normal( [128, n_classes] ))
            variable_summaries(weights)
        with tf.name_scope('output_biases'):
            biases = tf.Variable(tf.random_normal( [n_classes] ))
            variable_summaries(biases)
        output = tf.matmul(fc1, weights) + biases
        tf.summary.histogram('output', output)

    return output



with open('train1.txt', 'r') as train:
    train_lines = train.read().splitlines()


# Reads pfathes of images together with their labels

image_list, label_list = read_labeled_image_list('train1.txt')
t_image_list, t_label_list = read_labeled_image_list('test1.txt')

images = tf.convert_to_tensor(image_list, dtype=tf.string)
t_images = tf.convert_to_tensor(t_image_list, dtype=tf.string)

labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
t_labels = tf.convert_to_tensor(t_label_list, dtype=tf.int32)

'''labels = sess.run(labels)
    for i in range(10000,10020):
        print(labels[i])
sys.exit("Error message")'''

hm_epochs = 15
# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels],
                                            num_epochs=hm_epochs,
                                            shuffle=True)
t_input_queue = tf.train.slice_input_producer([t_images, t_labels],
                                            shuffle=True)

image, label = read_images_from_disk(input_queue)
t_image, t_label = read_images_from_disk(t_input_queue)

label = tf.one_hot(label, 47)
t_label = tf.one_hot(t_label, 47)

image = tf.cast(image, tf.float32)
image = tf.image.central_crop(image, .8125)
image = tf.image.resize_images(image, [52, 52])

with tf.Session() as sess:
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


t_image = tf.cast(t_image, tf.float32)
t_image = tf.image.central_crop(t_image, .8125)
t_image = tf.image.resize_images(t_image, [86, 86])

# Optional Image and Label Batching
#image_batch, label_batch = tf.train.batch([image, label],
#                                           batch_size=batch_size)

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size

image, label = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                      min_after_dequeue=min_after_dequeue)

t_image, t_label = tf.train.shuffle_batch([t_image, t_label], batch_size=batch_size, capacity=capacity,
                                      min_after_dequeue=min_after_dequeue)

x, y = image, label

with tf.Session() as sess:
    #threads = tf.train.start_queue_runners(sess)
    #init = tf.global_variables_initializer()
    #sess.run(init)
    #tf.train.start_queue_runners(sess)
    prediction = convolutional_neural_network(x)
    
    with tf.name_scope('training'):
        with tf.name_scope('cross_entropy'):
            cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction) )
        tf.summary.scalar('cross_entropy', cost)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer().minimize(cost)
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            with tf.name_scope('train_accuracy'):
                train_accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            tf.summary.scalar('train_accuracy', train_accuracy)


    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    sess.run(tf.local_variables_initializer())  #bitno
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)          #bitno

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('D:/train', sess.graph)
    
    #converting t_image from Tensor to numpy ndarray, for use in eval function for feed dict
    t_image, t_label = sess.run([t_image, t_label])
    
    hm_epochs = 10
    for epoch in range(hm_epochs):
        epoch_loss = 0

        n = len(train_lines)
        #n = int(mnist.train.num_examples/batch_size)
        n = int(n/batch_size)
        #print(n)
        for i in range(n):
            '''if i == 0:
            _y, _prediction = sess.run([y, prediction])
            with open('output.txt','wb') as output:
                np.savetxt(output, _y, delimiter=",")
                np.savetxt(output, _prediction, delimiter=",")
        '''
        #print(y)
            #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            if i % 10 == 0:
                summary, _, c = sess.run([merged, optimizer, cost])
                #summary, _, c = sess.run([merged, optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                train_writer.add_summary(summary, epoch * n + i)
                #break
            else:
                _, c = sess.run([optimizer, cost])
                #_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%H-%M')
        save_path = saver.save(sess, "D:/train/model"+st+".ckpt")
        print("Model saved in file: %s" % save_path)
        print('Epoch', epoch, 'completed out of', hm_epochs,'loss:', epoch_loss)

        #correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #test_accuracy = train_accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
        test_accuracy = train_accuracy.eval({x:t_image, y:t_label})

        print('Accuracy:', test_accuracy)

    #coord.request_stop()
    #coord.join(threads)