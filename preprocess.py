import os
import random

import tensorflow as tf
import numpy as np

# pre-processing parameters
DATA_DIR = 'data'
CATEGORIES = ['penguin', 'toothpaste', 'jacket']
NUM_LABELS = len(CATEGORIES)
NUM_EXAMPLES = 1000  # number of examples to read in per class
IMAGE_SIZE = 28  # set x, y pixel numbers for query/training/test examples

OUTPUT_DIR = 'tf_records'
OUTPUT_SHARDS = 4


# TODO: add back train/test split logic
# TODO: this may not scale well, consider using a file iterator
def read_bitmap(data_dir, categories, img_size):
    """Read in the data and return in as (img, class index) tuples"""

    category_filenames = []
    for catname in categories:
        filename = os.path.join(data_dir, catname + '.npy')
        category_filenames.append(filename)

    # holds tuples of img, class_index
    train_data = []

    # Read data and extract some for query data
    for i_category, category in enumerate(categories):
        data = np.load(category_filenames[i_category])
        n_data = len(data)
        n_categories = len(categories)  # number of classes
        print("[%d/%d] Reading category index %d: '%s' (%d images)" %
              (i_category + 1, n_categories, i_category, category, n_data))

        for j, data_j in enumerate(data):
            if j < NUM_EXAMPLES:
                # img = np.array(data_j).reshape((img_size, img_size))
                img = np.array(data_j)
                train_data.append((img, i_category))
            else:
                break

    return train_data


def create_bitmap_dataset(bitmap_data, output_dir, output_name, classnames, output_shards):
    """
    Writes the provided bitmap_data as tf.Example in tf.Record
    
    :param bitmap_data: tuple of lists containing (x_train, y_train, x_test, y_test)
    :param output_dir: path where to write the output.
    :param output_name: name to use when writing out records and classes
    :param classnames: array with classnames - is auto created if not passed in.
    :param output_shards: the number of shards to write the output in.
    :return: the class names as strings. classnames[classes[i]] is the
        textual representation of the class of the i-th data point.
    """

    def _pick_output_shard():
        return random.randint(0, output_shards - 1)

    writers = []
    for i in range(output_shards):
        writers.append(
            tf.python_io.TFRecordWriter("%s/%s-%05i-of-%05i" % (output_dir, output_name, i, output_shards)))

    for img, class_index in bitmap_data:
        features = {}
        features["class_index"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[class_index]))
        features["img"] = tf.train.Feature(float_list=tf.train.FloatList(value=img))
        f = tf.train.Features(feature=features)
        example = tf.train.Example(features=f)
        writers[_pick_output_shard()].write(example.SerializeToString())

    # Close all files
    for w in writers:
        w.close()

    # Write the class list.
    with tf.gfile.GFile(output_dir + "/" + output_name + ".classes", "w") as f:
        for class_name in classnames:
            f.write(class_name + "\n")
    return classnames

if __name__ == "__main__":
    bitmap_data = read_bitmap(DATA_DIR, CATEGORIES, IMAGE_SIZE)
    create_bitmap_dataset(bitmap_data, OUTPUT_DIR, "bitmap", CATEGORIES, OUTPUT_SHARDS)
