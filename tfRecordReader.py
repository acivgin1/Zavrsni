import io
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

filename = "mnist.tfrecords"
for serialized_example in tf.python_io.tf_record_iterator(filename):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    # traverse the Example format to get data
    image_raw = example.features.feature['image_raw'].bytes_list.value[0]
    label = example.features.feature['label'].int64_list.value[0]

    image = tf.decode_raw(image_raw, tf.float32)
    image = tf.reshape(image, (52, 52))
    # do something
    print(type(image))
    with tf.Session() as sess:
        image_raw = sess.run(image)
    print(image_raw.shape)
    plt.imshow(image_raw.squeeze(), cmap='gray')
    plt.show()
    print(image.shape)