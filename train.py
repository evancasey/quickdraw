import functools

import tensorflow as tf
import numpy as np

import preprocess
from model import create_conv2d_model

# model parameters
TRAIN_RECORD_PATTERN = 'tf_records/bit'
EVAL_RECORD_PATTERN = 'tf_records/'
MODEL_DIR = 'models'  # save/load conv2d model filename

# training parameters
SEED = 101
MAX_STEPS = 1000  # number of learning steps to train the model for
N_EPOCHS_CONV2D = 20  # number of epochs for conv2d training
BATCH_SIZE = 512  # batch size used in training
EVAL_BATCH_SIZE = 512  # batch size used in eval
LEARNING_RATE = 0.1


# def convert_img2norm(img_list, image_size):
#     norm_list = img_list.copy()
#     norm_list = norm_list.astype('float32') / 255
#     norm_list = np.reshape(norm_list, (len(norm_list), image_size, image_size, 1))
#     return norm_list


# def create_conv2d_placeholders():
#     # This is where training samples and labels are fed to the graph.
#     # These placeholder nodes will be fed a batch of training data at each
#     # training step using the {feed_dict} argument to the Run() call below.
#     train_data_node = tf.placeholder(
#         tf.float32,
#         shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
#     train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
#     eval_data = tf.placeholder(
#         tf.float32,
#         shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))
#
#     return train_data_node, train_labels_node, eval_data


def get_input_fn(mode, tfrecord_pattern, batch_size):
    """Creates an input_fn that stores all the data in memory."""

    def _parse_tfexample_fn(example_proto, mode):
        """Parse a single record which is expected to be a tensorflow.Example."""
        feature_to_type = {
            "img": tf.FixedLenFeature([preprocess.IMAGE_SIZE * preprocess.IMAGE_SIZE], dtype=tf.float32),
            # TODO: also parse conv1d features
        }
        if mode != tf.estimator.ModeKeys.PREDICT:
            # The labels won't be available at inference time, so don't add them
            # to the list of feature_columns to be read.
            feature_to_type["class_index"] = tf.FixedLenFeature([1], dtype=tf.int64)

        parsed_features = tf.parse_single_example(example_proto, feature_to_type)
        labels = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = parsed_features["class_index"]
        return parsed_features, labels

    def _input_fn():
        """
        Estimator `input_fn`.
        
        :return: A tuple of:
          - Dictionary of string feature name to `Tensor`.
          - `Tensor` of target labels.
        """
        dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.repeat()
        # Preprocesses 10 files concurrently and interleaves records from each file.
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1)
        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, mode=mode),
            num_parallel_calls=10)
        dataset = dataset.prefetch(10000)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1000000)
        # Our inputs are variable length, so pad them.
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=dataset.output_shapes)
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    return _input_fn


def model_fn(features, labels, mode, params):
    model = create_conv2d_model(features, params)
    image = features  # TODO: do feature norm
    if isinstance(image, dict):
        image = features['image']
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        logits = model(image, training=True)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))
        # Name the accuracy tensor 'train_accuracy' to demonstrate the LoggingTensorHook.
        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
    else:
        # TODO: support eval/train modes
        raise NotImplementedError


def create_estimator_and_specs(run_config):
    """Creates an Experiment configuration based on the estimator and input fn."""

    # add more params here as needed
    model_params = tf.contrib.training.HParams(
        batch_size=BATCH_SIZE,
        num_classes=preprocess.NUM_LABELS,
        learning_rate=LEARNING_RATE)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)

    train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.TRAIN,
        tfrecord_pattern=TRAIN_RECORD_PATTERN,
        batch_size=BATCH_SIZE), max_steps=MAX_STEPS)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.EVAL,
        tfrecord_pattern=EVAL_RECORD_PATTERN,
        batch_size=EVAL_BATCH_SIZE))

    return estimator, train_spec, eval_spec


if __name__ == "__main__":
    estimator, train_spec, eval_spec = create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            model_dir=MODEL_DIR,
            save_checkpoints_secs=300,
            save_summary_steps=100))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
