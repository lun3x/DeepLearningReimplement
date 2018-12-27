############################################################
#                                                          #
#  Code for Lab 1: Intro to TensorFlow and Blue Crystal 4  #
#                                                          #
############################################################

'''Based on TensorFLow's tutorial: A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import os
import os.path

import tensorflow as tf

import pickle

from utils import GZTan

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 10, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-3, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-width', 80, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 80, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')


run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_BN_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                        lr=FLAGS.learning_rate))

# the initialiser object implementing Xavier initialisation
# we will generate weights from the uniform distribution
xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

def deepnn(x, test_flag):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.

  Args:
      x: an input tensor with the dimensions (N_examples, 3072), where 3072 is the
        number of pixels in a standard CIFAR10 image.

  Returns:
      y: is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the object images into one of 10 classes
        (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
      img_summary: a string tensor containing sampled input images.
    """
    # Reshape to use within a convolutional neural net.  Last dimension is for
    # 'features' - it would be 1 one for a grayscale image, 3 for an RGB image,
    # 4 for RGBA, etc.

    #-1 is used here to infer the shape. We might not know how many N_examples are passed in as x
    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height])
    #delta = tf.random_uniform([], minval=0, maxval=0.9, dtype=tf.float32, seed=None, name=None)
    # x_image = tf.cond(tf.equal(test_flag[0], 0), lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(img), x_image), lambda: x_image)
    # x_image = tf.map_fn(melspectrogram, x_image)
    #scale = tf.random_uniform([], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None, name=None)
    #new_width = tf.cast(FLAGS.img_width * scale, tf.int32)
    #new_height = tf.cast(FLAGS.img_height * scale, tf.int32)
    #augmented_image = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, new_height, new_width), x_image)

    # x_spectrogram = melspectrogram(x_image)

    img_summary = tf.summary.image('Input_images', x_image)

    # First convolutional layer - maps one image to 16 feature maps.
    with tf.variable_scope('Conv_Spectral_1'):
        conv_spec_1 = tf.layers.conv2d(
            inputs=x_image,
            filters=16,
            kernel_size=[10, 23],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_spec_1'
        )

        # MAYBE NEED TO DO PROPER WEIGHTING AND BIAS HERE???
        conv_spec_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_spec_1, name='conv_spec_1_bn', training=(not test_flag[0])))

        # Pooling layer - downsamples by 2X.
        pool_spec_1 = tf.layers.max_pooling2d(
            inputs=conv_spec_1_bn,
            pool_size=[1, 20],
            strides=[1, 1, 20],
            name='pool1'
        )

        # MAYBE RESHAPE???
        reshaped_pool_spec_1 = tf.reshape(pool_spec_1, [-1, 5120])

    with tf.variable_scope('Conv_Temporal_1'):
        conv_temp_1 = tf.layers.conv2d(
            inputs=x_image,
            filters=16,
            kernel_size=[21, 20],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_spec_1'
        )

        # MAYBE NEED TO DO PROPER WEIGHTING AND BIAS HERE???
        conv_temp_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_temp_1, name='conv_temp_1_bn', training=(not test_flag[0])))

        # Pooling layer - downsamples by 2X.
        pool_temp_1 = tf.layers.max_pooling2d(
            inputs=conv_temp_1_bn,
            pool_size=[20, 1],
            strides=[1, 20, 1],
            name='pool1'
        )

        # MAYBE RESHAPE???
        reshaped_pool_temp_1 = tf.reshape(pool_temp_1, [-1, 5120])

    with tf.variable_scope('Merge'):
        merged_streams = tf.concat([reshaped_pool_spec_1, reshaped_pool_temp_1], 1) # maybe 0?

    with tf.variable_scope('FC_1'):
        fc1 = tf.layers.dense(
            inputs=merged_streams,
            units=200,
            kernel_initializer=xavier_initializer,
            name='fc1'
        )

        fc1_dropout = tf.layers.dropout(
            fc1,
            training=(not test_flag[0]), # not training
            name='fc1_dropout'
        )

        #Fully connected network
        # W_fc1 = weight_variable([4096, 1024])
        # b_fc1 = bias_variable([1024])
        # h_fc1 = tf.nn.relu(tf.matmul(reshapedh_pool2, W_fc1) + b_fc1)

    # with tf.variable_scope('FC_2'):
    #     W_fc2 = weight_variable([1024, 1024])
    #     b_fc2 = bias_variable([1024])
    #     h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.variable_scope('FC_3'):
        fc3 = tf.layers.dense(
            inputs=fc1_dropout,
            units=FLAGS.num_classes,
            kernel_initializer=xavier_initializer,
            name='fc3'
        )
        return fc3, img_summary


def main(_):
    tf.reset_default_graph()

    # Import data
    gztan = GZTan(FLAGS.batch_size)

    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
        test_flag = tf.placeholder(tf.uint8, [1])

    # Build the graph for the deep net
    y_conv, img_summary = deepnn(x, test_flag)

    # Define your loss function - softmax_cross_entropy
    with tf.variable_scope("x_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    # Define your AdamOptimiser, using FLAGS.learning_rate to minimixe the loss function
    # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    batch_number = tf.Variable(0, trainable=False)
    our_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch_number, 1000, 0.8)
    optimizer = tf.train.AdamOptimizer(our_learning_rate).minimize(cross_entropy, global_step=batch_number)
    # calculate the prediction and the accuracy
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)

    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([img_summary, acc_summary])
    training_summary = tf.summary.merge([img_summary, loss_summary])
    test_summary = tf.summary.merge([img_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph, flush_secs=5)

        sess.run(tf.global_variables_initializer())

        # Training and validation
        for step in range(FLAGS.max_steps):
            # Training: Backpropagation using train set
            # (trainImages, trainLabels) = cifar.getTrainBatch()
            # (testImages, testLabels) = cifar.getTestBatch()
            (train_samples, train_labels) = gztan.get_train_batch()
            (test_samples, test_labels) = gztan.get_test_batch()

            _, summary_str = sess.run([optimizer, training_summary], feed_dict={x: train_samples, test_flag:[0], y_: train_labels})

            if step % (FLAGS.log_frequency + 1) == 0:
                summary_writer.add_summary(summary_str, step)

            # Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                validation_accuracy, summary_str = sess.run([accuracy, validation_summary], feed_dict={x: test_samples, test_flag:[1], y_: test_labels})
                print('step %d, accuracy on validation batch: %g' % (step, validation_accuracy))
                summary_writer_validation.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Testing

        # resetting the internal batch indexes
        # cifar.reset()
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

        # don't loop back when we reach the end of the test set
        while evaluated_images != len(gztan.test_samples):
            (test_samples, test_labels) = gztan.get_test_batch()
            test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x: test_samples, test_flag:[1], y_: test_labels})

            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
            evaluated_images = evaluated_images + test_labels.shape[0]

        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)



if __name__ == '__main__':
    tf.app.run(main=main)
