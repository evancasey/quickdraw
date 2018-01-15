import tensorflow as tf


def create_conv2d_model(features, params):
    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.global_variables_initializer().run()}
    conv1_weights = tf.Variable(  # 3x3 filter, depth 32
        tf.truncated_normal([3, 3, 1, 32], stddev=0.1, seed=params.seed, dtype=tf.float32))
    conv1_biases = tf.Variable(
        tf.zeros([32], dtype=tf.float32))
    conv2_weights = tf.Variable(
        tf.truncated_normal([3, 3, 32, 64], stddev=0.1, seed=params.seed, dtype=tf.float32))
    conv2_biases = tf.Variable(
        tf.constant(0.1, shape=[32], dtype=tf.float32))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([params.image_size // 4 * params.image_size // 4 * 64, 128], stddev=0.1, seed=params.seed,
                            dtype=tf.float32))
    fc1_biases = tf.Variable(
        tf.constant(0.1, shape=[512], dtype=tf.float32))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, params.num_labels], stddev=0.1, seed=params.seed, dtype=tf.float32))
    fc2_biases = tf.Variable(
        tf.constant(0.1, shape=[params.num_labels], dtype=tf.float32))

    img_data = tf.reshape(features["img"], [28, 28])
    model = conv2d_model(img_data, conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases,
                         fc2_weights, fc2_biases, params.seed, train=False)

    return model


# TODO: try out channels_first for perf boost
# TODO: consider using layers API
def conv2d_model(features,
                 conv1_weights,
                 conv1_biases,
                 conv2_weights,
                 conv2_biases,
                 fc1_weights,
                 fc1_biases,
                 fc2_weights,
                 fc2_biases,
                 seed,
                 train=False):
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(features,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=seed)
    return tf.matmul(hidden, fc2_weights) + fc2_biases