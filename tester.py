import tensorflow as tf

with tf.Session() as sess:
    input = tf.Variable(tf.zeros([1, 2, 2, 1]))
    window = tf.Variable(tf.random_normal([2, 2, 1, 2]))
    biases = tf.Variable(tf.random_normal([1, 2, 2, 2]))
    preact1 = tf.nn.conv2d(input, window, [1, 1, 1, 1], 'SAME')
    preact2 = preact1 + biases
    preact3 = tf.nn.max_pool(preact2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    tf.global_variables_initializer().run()


    #print('input: {}'.format(input.shape))
    print('windows: {}'.format(window.shape))
    print('biases: {}'.format(biases.shape))
    print('conv: {}'.format(preact1.shape))
    #print('conv + bias: {}'.format(preact2.shape))
    print('maxpool: {}'.format(preact3.shape))
    a, b, c, d, e, f = sess.run([input, window, biases, preact1, preact2, preact3])

    # print(a)
    print(c)
    print(e)
    print(f)
