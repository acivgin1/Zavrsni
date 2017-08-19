import tensorflow as tf
import datetime
import time
import os.path

from tfLoader import tfrecords_loader
from tfHelperFunctions import conv_layer
from tfHelperFunctions import max_pool_layer
from tfHelperFunctions import nn_layer
from tfHelperFunctions import variable_summaries

from tensorflow.python.client import timeline

# import sys
# import numpy as np
# import matplotlib.pyplot as plt

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_classes = 47
batch_size = 2048
hm_epochs = 10

# placeholders used only when loading non Tensor data, like in the mnist example
# x = tf.placeholder(tf.float32, [None, 52, 52, 1])
# y = tf.placeholder(tf.float32, [None, n_classes])


def convolutional_neural_network(data):
    data = tf.reshape(data, shape=[-1, 52, 52, 1], name='input')

    # conv_layer(data, filter_height, filter_width, in_channels, out_channels, strides, padding, layer_name)
    conv1 = conv_layer(data, 5, 5, 1, 8, [1, 1, 1, 1], 'SAME', 'conv1')
    # max_pool_layer(data, ksize, strides, padding, layer_name)
    conv1 = max_pool_layer(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', 'max_pool1')

    conv2 = conv_layer(conv1, 5, 5, 8, 32, [1, 1, 1, 1], 'SAME', 'conv2')
    conv2 = max_pool_layer(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', 'max_pool2')

    # conv3 = conv_layer(conv2, 5, 5, 32, 16, [1,1,1,1], 'SAME', 'conv3')
    # conv3 = max_pool_layer(conv3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', 'max_pool3')

    shape_dim = 13*13*32
    fc1 = tf.reshape(conv2, shape=[-1, shape_dim], name='conv2_maxpool3')
    fc1 = nn_layer(fc1, shape_dim, 128, 'fully_connected1')
    # fc1 = tf.nn.dropout(fc1, 0.85)
    # fc2 = nn_layer(fc1, 1024, 128, 'fully_connected2')

    with tf.name_scope('output'):
        with tf.name_scope('output_weights'):
            weights = tf.Variable(tf.random_normal([128, n_classes]), name='output' + 'weights')
            variable_summaries(weights)
        with tf.name_scope('output_biases'):
            biases = tf.Variable(tf.random_normal([n_classes]), name='output' + 'biases')
            variable_summaries(biases)
        output = tf.matmul(fc1, weights) + biases
        tf.summary.histogram('output', output)
    return output


def main():
    x, y, t_image, t_label, n = tfrecords_loader(n_classes, batch_size, hm_epochs)
    # t_image, t_label = zip_loader('D:/by_merge.zip', n_classes, batch_size, load_train=False)
    # _, _, train_lines = zip_loader('D:/by_merge.zip', n_classes, batch_size, load_train=True)

    with tf.Session() as sess:
        prediction = convolutional_neural_network(x)

        # defining name scopes for cost and accuracy to be written into the summary file
        with tf.name_scope('training'):
            with tf.name_scope('cross_entropy'):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
            tf.summary.scalar('cross_entropy', cost)

            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer().minimize(cost) # using Adam Optimizer algorithm for reducing cost

            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                with tf.name_scope('train_accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                tf.summary.scalar('train_accuracy', accuracy)

        if os.path.isfile("D:/train/current/saves/model.ckpt.meta"):
            saver = tf.train.Saver()
            saver.restore(sess, "D:/train/current/saves/model.ckpt")
            print("Model restored.")
        else:
            saver = tf.train.Saver()
            print("Previous save missing...\nStarting with random values.")
            tf.global_variables_initializer().run()

        tf.local_variables_initializer().run()  # loads images into x and y variables

        coord = tf.train.Coordinator()  # coordinator used for coordinating between various threads that load data
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)          # bitno

        # converting t_image from Tensor to numpy ndarray, for use in eval function for feed dict
        # t_image, t_label = sess.run([t_image, t_label])

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('D:/train/current', sess.graph)

        for epoch in range(hm_epochs):
            epoch_loss = 0

            #n = len(train_lines)
            # n = int(mnist.train.num_examples/batch_size)
            n = int(n/batch_size)

            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()

            for i in range(100):
                # epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # epoch_x, epoch_y, _ = tf_loader('D:/by_merge.zip', n_classes, batch_size, load_train=True, \
                #     current=i*batch_size)
                if i % 5 == 0:
                    # summary, _, c = sess.run([merged, optimizer, cost], options=run_options,
                    #                          run_metadata=run_metadata)
                    summary, _, c = sess.run([merged, optimizer, cost])

                    # summary, _, c = sess.run([merged, optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    train_writer.add_summary(summary, (epoch * n + i)/10)

                    # Create the Timeline object, and write it to a json
                    # tl = timeline.Timeline(run_metadata.step_stats)
                    # ctf = tl.generate_chrome_trace_format()
                    # with open('timeline' + str(i) + '.json', 'w') as f:
                    #     f.write(ctf)
                else:
                    _, c = sess.run([optimizer, cost])
                    # _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            # Creating saves after an epoch
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%H-%M')
            new_path = 'D:/train/current/saves'
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            save_path = saver.save(sess, new_path + "/model" + st + ".ckpt")
            saver.save(sess, new_path + "/model.ckpt")

            print("Model saved in file: %s" % save_path)
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            # Added output of current Confusion Matrix
            conf_matrix = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(prediction, 1), num_classes=n_classes)

            conf_matrix_eval, test_accuracy = sess.run([conf_matrix, accuracy])  # , feed_dict={x: t_image, y: t_label})
            with open('D:/train/current/confMatrix{}.txt'.format(epoch), 'w') as confMatrixOutput:
                for line in conf_matrix_eval:
                    for word in line:
                        confMatrixOutput.write('{:>4}'.format(word))
                        print('{:>4}'.format(word), end='')
                    print('')
                    confMatrixOutput.write('\n')
            # Print out the current test accuracy
            print('Accuracy:', test_accuracy)

        coord.request_stop()
        coord.join(threads)

main()