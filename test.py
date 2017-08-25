import tensorflow as tf
import datetime
import time
import os.path

from tqdm import tqdm
from tfLoader import tfrecords_loader
from tfHelperFunctions import conv_layer
from tfHelperFunctions import max_pool_layer
from tfHelperFunctions import nn_layer
from tfHelperFunctions import variable_summaries

from tensorflow.python.client import timeline
import numpy as np
import sys
# import matplotlib.pyplot as plt

n_classes = 47
batch_size = 2048
hm_epochs = 10
current_location = "current"
beta = 0.01

# 1280537 parameters


def convolutional_neural_network(data):
    data = tf.reshape(data, shape=[-1, 52, 52, 1], name='input')

    # conv_layer(data, filter_height, filter_width, in_channels, out_channels, strides, padding, layer_name)
    conv1 = conv_layer(data, 5, 5, 1, 40, [1, 1, 1, 1], 'SAME', 'conv1')
    # max_pool_layer(data, ksize, strides, padding, layer_name)
    conv1 = max_pool_layer(conv1, [1, 4, 4, 1], [1, 3, 3, 1], 'SAME', 'max_pool1')

    conv2 = conv_layer(conv1, 5, 5, 40, 50, [1, 1, 1, 1], 'SAME', 'conv2')
    conv2 = max_pool_layer(conv2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', 'max_pool2')

    #conv3 = conv_layer(conv2, 5, 5, 32, 32, [1,1,1,1], 'SAME', 'conv3')
    #conv3 = max_pool_layer(conv3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', 'max_pool3')

    shape_dim = 9*9*50
    fc1 = tf.reshape(conv2, shape=[-1, shape_dim], name='conv2_maxpool3')
    fc1 = nn_layer(fc1, shape_dim, 300, 'fully_connected1', act=tf.nn.relu)
    # fc2 = nn_layer(fc1, 1024, 128, 'fully_connected2')

    with tf.name_scope('output'):
        with tf.name_scope('output_weights'):
            weights = tf.Variable(tf.random_normal([300, n_classes]), name='output' + 'weights')
            variable_summaries(weights)
        with tf.name_scope('output_biases'):
            biases = tf.Variable(tf.random_normal([n_classes]), name='output' + 'biases')
            variable_summaries(biases)
        output = tf.matmul(fc1, weights) + biases
        tf.summary.histogram('output', output)
    return output


def main():
    print('Starting training.')
    x, y, num_examples = tfrecords_loader(n_classes, batch_size, hm_epochs)

    with tf.Session() as sess:
        prediction = convolutional_neural_network(x)
        weights_fc = [v for v in tf.global_variables() if v.name ==
                      "fully_connected1/weights/fully_connected1weights:0"][0]
        weights_fl = [v for v in tf.global_variables() if v.name ==
                      "output/output_weights/outputweights:0"][0]
        print(weights_fc.shape)
        print(weights_fl.shape)
        # defining name scopes for cost and accuracy to be written into the summary file
        with tf.name_scope('training'):
            with tf.name_scope('cross_entropy'):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
                regularizer_fc = tf.nn.l2_loss(weights_fc)
                regularizer_fl = tf.nn.l2_loss(weights_fl)
                cost = tf.reduce_mean(cost + beta * (regularizer_fc + regularizer_fl))
            tf.summary.scalar('cross_entropy', cost)
            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer().minimize(cost)  # using Adam Optimizer algorithm for reducing cost
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                with tf.name_scope('train_accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                tf.summary.scalar('train_accuracy', accuracy)

        if os.path.isfile(current_location + "/saves/model.ckpt.meta"):
            saver = tf.train.Saver()
            saver.restore(sess, current_location + "/saves/model.ckpt")
            print("Model restored.")
        else:
            saver = tf.train.Saver()
            print("Previous save missing...\nStarting with random values.")
            tf.global_variables_initializer().run()

        tf.local_variables_initializer().run()  # loads images into x and y variables
        coord = tf.train.Coordinator()  # coordinator used for coordinating between various threads that load data
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)          # bitno

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(current_location, sess.graph)

        for epoch in range(hm_epochs):
            epoch_loss = 0
            n = int(num_examples/batch_size)

            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()

            for i in tqdm(range(n)):
                if i % 5 == 0:
                    # summary, _, c = sess.run([merged, optimizer, cost], options=run_options,
                    #                          run_metadata=run_metadata)
                    summary, _, c = sess.run([merged, optimizer, cost])
                    train_writer.add_summary(summary, (epoch * n + i)/5)
                    # Create the Timeline object, and write it to a json
                    # tl = timeline.Timeline(run_metadata.step_stats)
                    # ctf = tl.generate_chrome_trace_format()
                    # with open('timeline' + str(i) + '.json', 'w') as f:
                    #     f.write(ctf)
                else:
                    _, c = sess.run([optimizer, cost])
                epoch_loss += c

            # Creating saves after an epoch
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%H-%M')
            new_path = current_location + "/saves"
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            save_path = saver.save(sess, new_path + "/model" + st + ".ckpt")
            saver.save(sess, new_path + "/model.ckpt")

            print("Model saved in file: %s" % save_path)
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            # Added output of current Confusion Matrix
            conf_matrix = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(prediction, 1), num_classes=n_classes)

            conf_matrix_eval, test_accuracy = sess.run([conf_matrix, accuracy])  # , feed_dict={x: t_image, y: t_label})
            with open(current_location + "/confMatrix{}.txt".format(epoch), 'w') as confMatrixOutput:
                for line in conf_matrix_eval:
                    for word in line:
                        confMatrixOutput.write('{:>4}'.format(word))
                    confMatrixOutput.write('\n')
            # Print out the current test accuracy
            print('Accuracy:', test_accuracy)

        coord.request_stop()
        coord.join(threads)


def test():
    print('Starting testing and confusion matrix and accuracy evaluation.')
    batch_size = 2048
    hm_epochs = 1
    x, y, n = tfrecords_loader(n_classes, batch_size, hm_epochs, train=False)
    prediction = convolutional_neural_network(x)

    with tf.Session() as sess:
        if os.path.isfile(current_location + "/saves/model.ckpt.meta"):
            saver = tf.train.Saver()
            saver.restore(sess, current_location + "/saves/model.ckpt")
            print("Model restored.")
        else:
            print("Save missing.\nExiting...")
            sys.exit()

        tf.local_variables_initializer().run()  # loads images into x and y variables
        coord = tf.train.Coordinator()  # coordinator used for coordinating between various threads that load data
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # Added output of current Confusion Matrix
        conf_matrix = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(prediction, 1), num_classes=n_classes)
        conf_matrix_eval = np.zeros((1, 47, 47))

        for _ in tqdm(range(int(n/batch_size)+1)):
            current_conf_matrix_eval = sess.run([conf_matrix])
            conf_matrix_eval = np.add(conf_matrix_eval, np.asarray(current_conf_matrix_eval))

        with open(current_location + "/confMatrixTest.txt", 'w') as confMatrixOutput:
            for i in range(47):
                for j in range(47):
                    confMatrixOutput.write('{:>5}'.format(int(conf_matrix_eval[(0, i, j)])))
                confMatrixOutput.write('\n')
        diagonal_sum = 0
        for i in range(47):
            diagonal_sum = diagonal_sum + conf_matrix_eval[(0, i, i)]
        accuracy = 100 * diagonal_sum / np.sum(conf_matrix_eval)
        print('Accuracy: {} of {} examples'.format(accuracy, int(np.sum(conf_matrix_eval))))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
    #test()