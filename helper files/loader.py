import tensorflow as tf

'''
# step 2
filename_queue = tf.train.string_input_producer(lines)

# step 3: read, decode and resize images
reader = tf.WholeFileReader()
filename, content = reader.read(filename_queue)
image = tf.image.decode_png(content, channels=1)
image = tf.cast(image, tf.float32)
resized_image = tf.image.resize_images(image, [32, 32])
# step 4: Batching
image_batch = tf.train.batch([resized_image], batch_size=8)
'''
    
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
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label


with open('train.txt','r') as input1:
    lines = input1.read().splitlines()
    # Reads pfathes of images together with their labels
    image_list, label_list = read_labeled_image_list(lines)
    
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)
    
    image, label = read_images_from_disk(input_queue)
    
    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    image = tf.image.preprocess_image(image)
    label = tf.image.preprocess_label(label)
    
    # Optional Image and Label Batching
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size)