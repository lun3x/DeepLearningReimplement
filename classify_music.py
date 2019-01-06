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
import numpy as np
import librosa
from librosa.display import specshow
import pickle
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab

# from utils import GZTan
from gztan_utils import GZTan2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('net-depth', 'shallow',
                            'Whether to use the deep or shallow network. (default: %(default)s)')
tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 1,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
# tf.app.flags.DEFINE_integer('batch-size', 128, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-batches', 750, 'Number of mini-batches (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 5e-5, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-width', 80, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 80, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/{d}/logs/'.format(cwd=os.getcwd(), d='shallow_100'),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')


run_log_dir = os.path.join(FLAGS.log_dir,
                           'genre_classify')

# the initialiser object implementing Xavier initialisation
# we will generate weights from the uniform distribution
xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

def shallownn(x, train_flag):
    """shallownn builds the graph for a deep net for classifying CIFAR10 images.

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
    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, 1])

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
        # conv_spec_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_spec_1, name='conv_spec_1_bn', training=train_flag[0]))
        h_conv_spec_1 = tf.nn.relu(conv_spec_1, name='conv_spec_1_relu')

        # Pooling layer - downsamples by 2X.
        pool_spec_1 = tf.layers.max_pooling2d(
            inputs=h_conv_spec_1,
            pool_size=[1, 20],
            strides=[1, 20],
            name='pool_spec_1'
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
        # conv_temp_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_temp_1, name='conv_temp_1_bn', training=train_flag[0]))
        h_conv_temp_1 = tf.nn.relu(conv_temp_1, name='conv_temp_1_relu')

        # Pooling layer - downsamples by 2X.
        pool_temp_1 = tf.layers.max_pooling2d(
            inputs=h_conv_temp_1,
            pool_size=[20, 1],
            strides=[20, 1],
            name='pool_temp_1'
        )

        # MAYBE RESHAPE???
        reshaped_pool_temp_1 = tf.reshape(pool_temp_1, [-1, 5120])

    with tf.variable_scope('Merge'):
        merged_streams = tf.concat([reshaped_pool_spec_1, reshaped_pool_temp_1], 1, name='merged_streams') # maybe 0?

        merged_dropout = tf.layers.dropout(
            merged_streams,
            rate=0.1,
            training=train_flag[0], # not training
            name='merged_dropout'
        )

    with tf.variable_scope('FC_1'):
        fc1 = tf.layers.dense(
            inputs=merged_dropout,
            units=200,
            kernel_initializer=xavier_initializer,
            name='fc1'
        )

    with tf.variable_scope('FC_3'):
        fc3 = tf.layers.dense(
            inputs=fc1,
            units=FLAGS.num_classes,
            kernel_initializer=xavier_initializer,
            name='fc3'
        )
        return fc3, img_summary

def deepnn(x, train_flag):
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
    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, 1])

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
        # conv_spec_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_spec_1, name='conv_spec_1_bn', training=train_flag[0]))
        h_conv_spec_1 = tf.nn.relu(conv_spec_1, name='conv_spec_1_relu')

        # Pooling layer - downsamples by 2X.
        pool_spec_1 = tf.layers.max_pooling2d(
            inputs=h_conv_spec_1,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_spec_1'
        )

        # MAYBE RESHAPE???
        # reshaped_pool_spec_1 = tf.reshape(pool_spec_1, [-1, 5120])

    with tf.variable_scope('Conv_Spectral_2'):
        conv_spec_2 = tf.layers.conv2d(
            inputs=pool_spec_1,
            filters=32,
            kernel_size=[5, 11],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_spec_2'
        )

        # MAYBE NEED TO DO PROPER WEIGHTING AND BIAS HERE???
        # conv_spec_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_spec_1, name='conv_spec_1_bn', training=train_flag[0]))
        h_conv_spec_2 = tf.nn.relu(conv_spec_2, name='conv_spec_2_relu')

        # Pooling layer - downsamples by 2X.
        pool_spec_2 = tf.layers.max_pooling2d(
            inputs=h_conv_spec_2,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_spec_2'
        )

        # MAYBE RESHAPE???
        # reshaped_pool_spec_1 = tf.reshape(pool_spec_1, [-1, 5120])

    with tf.variable_scope('Conv_Spectral_3'):
        conv_spec_3 = tf.layers.conv2d(
            inputs=pool_spec_2,
            filters=64,
            kernel_size=[3, 5],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_spec_3'
        )

        # MAYBE NEED TO DO PROPER WEIGHTING AND BIAS HERE???
        # conv_spec_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_spec_1, name='conv_spec_1_bn', training=train_flag[0]))
        h_conv_spec_3 = tf.nn.relu(conv_spec_3, name='conv_spec_3_relu')

        # Pooling layer - downsamples by 2X.
        pool_spec_3 = tf.layers.max_pooling2d(
            inputs=h_conv_spec_3,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_spec_3'
        )

        # MAYBE RESHAPE???
        # reshaped_pool_spec_1 = tf.reshape(pool_spec_1, [-1, 5120])

    with tf.variable_scope('Conv_Spectral_4'):
        conv_spec_4 = tf.layers.conv2d(
            inputs=pool_spec_3,
            filters=128,
            kernel_size=[2, 4],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_spec_4'
        )

        # MAYBE NEED TO DO PROPER WEIGHTING AND BIAS HERE???
        # conv_spec_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_spec_1, name='conv_spec_1_bn', training=train_flag[0]))
        h_conv_spec_4 = tf.nn.relu(conv_spec_4, name='conv_spec_4_relu')

        # Pooling layer - downsamples by 2X.
        pool_spec_4 = tf.layers.max_pooling2d(
            inputs=h_conv_spec_4,
            pool_size=[1, 5],
            strides=[1, 5],
            name='pool_spec_4'
        )

        # MAYBE RESHAPE???
        reshaped_pool_spec_4 = tf.reshape(pool_spec_4, [-1, 2560])


    with tf.variable_scope('Conv_Temporal_1'):
        conv_temp_1 = tf.layers.conv2d(
            inputs=x_image,
            filters=16,
            kernel_size=[21, 10],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_temp_1'
        )

        # MAYBE NEED TO DO PROPER WEIGHTING AND BIAS HERE???
        # conv_temp_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_temp_1, name='conv_temp_1_bn', training=train_flag[0]))
        h_conv_temp_1 = tf.nn.relu(conv_temp_1, name='conv_temp_1_relu')

        # Pooling layer - downsamples by 2X.
        pool_temp_1 = tf.layers.max_pooling2d(
            inputs=h_conv_temp_1,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_temp_1'
        )

        # MAYBE RESHAPE???
        # reshaped_pool_temp_1 = tf.reshape(pool_temp_1, [-1, 5120])

    with tf.variable_scope('Conv_Temporal_2'):
        conv_temp_2 = tf.layers.conv2d(
            inputs=pool_temp_1,
            filters=32,
            kernel_size=[10, 5],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_temp_2'
        )

        # MAYBE NEED TO DO PROPER WEIGHTING AND BIAS HERE???
        # conv_temp_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_temp_1, name='conv_temp_1_bn', training=train_flag[0]))
        h_conv_temp_2 = tf.nn.relu(conv_temp_2, name='conv_temp_2_relu')

        # Pooling layer - downsamples by 2X.
        pool_temp_2 = tf.layers.max_pooling2d(
            inputs=h_conv_temp_2,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_temp_2'
        )

        # MAYBE RESHAPE???
        # reshaped_pool_temp_1 = tf.reshape(pool_temp_1, [-1, 5120])

    with tf.variable_scope('Conv_Temporal_3'):
        conv_temp_3 = tf.layers.conv2d(
            inputs=pool_temp_2,
            filters=64,
            kernel_size=[5, 3],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_temp_3'
        )

        # MAYBE NEED TO DO PROPER WEIGHTING AND BIAS HERE???
        # conv_temp_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_temp_1, name='conv_temp_1_bn', training=train_flag[0]))
        h_conv_temp_3 = tf.nn.relu(conv_temp_3, name='conv_temp_3_relu')

        # Pooling layer - downsamples by 2X.
        pool_temp_3 = tf.layers.max_pooling2d(
            inputs=h_conv_temp_3,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_temp_3'
        )

        # MAYBE RESHAPE???
        # reshaped_pool_temp_1 = tf.reshape(pool_temp_1, [-1, 5120])

    with tf.variable_scope('Conv_Temporal_4'):
        conv_temp_4 = tf.layers.conv2d(
            inputs=pool_temp_3,
            filters=128,
            kernel_size=[4, 2],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_temp_4'
        )

        # MAYBE NEED TO DO PROPER WEIGHTING AND BIAS HERE???
        # conv_temp_1_bn = tf.nn.relu(tf.layers.batch_normalization(conv_temp_1, name='conv_temp_1_bn', training=train_flag[0]))
        h_conv_temp_4 = tf.nn.relu(conv_temp_4, name='conv_temp_4_relu')

        # Pooling layer - downsamples by 2X.
        pool_temp_4 = tf.layers.max_pooling2d(
            inputs=h_conv_temp_4,
            pool_size=[5, 1],
            strides=[5, 1],
            name='pool_temp_4'
        )

        # MAYBE RESHAPE???
        reshaped_pool_temp_4 = tf.reshape(pool_temp_4, [-1, 2560])

    with tf.variable_scope('Merge'):
        merged_streams = tf.concat([reshaped_pool_spec_4, reshaped_pool_temp_4], 1, name='merged_streams') # maybe 0?

        merged_dropout = tf.layers.dropout(
            merged_streams,
            rate=0.25,
            training=train_flag[0], # not training
            name='merged_dropout'
        )

    with tf.variable_scope('FC_1'):
        fc1 = tf.layers.dense(
            inputs=merged_dropout,
            units=200,
            kernel_initializer=xavier_initializer,
            name='fc1'
        )

        h_fc1 = tf.nn.relu(fc1, name='fc1_relu')

    with tf.variable_scope('FC_3'):
        fc3 = tf.layers.dense(
            inputs=h_fc1,
            units=FLAGS.num_classes,
            kernel_initializer=xavier_initializer,
            name='fc3'
        )
        return fc3, img_summary

def raw_acc(y_, y_conv):
    raw_correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
    return tf.reduce_mean(tf.cast(raw_correct_prediction, tf.float32))

def main(_):
    tf.reset_default_graph()

    # Import data
    gztan = GZTan2(FLAGS.num_batches)

    print('num train tracks: {}'.format(gztan.nTrainTracks))

    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, 80, 80, 1])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
        train_flag = tf.placeholder(tf.bool, [1])
        label = tf.placeholder(tf.int32, [FLAGS.num_classes])

    # Build the graph for the deep net
    if FLAGS.net_depth == 'shallow':
        print('SHALLOW')
        y_conv, img_summary = shallownn(x, train_flag)
    elif FLAGS.net_depth == 'deep':
        print('DEEP')
        y_conv, img_summary = deepnn(x, train_flag)
    else:
        print("Error: Unrecognised depth.")
        return

    # Define your loss function - softmax_cross_entropy
    with tf.variable_scope("x_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        # l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0001)
        # weights = tf.trainable_variables()
        # regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        # regularized_cross_entropy = cross_entropy + regularization_penalty

    # Define your AdamOptimiser, using FLAGS.learning_rate to minimixe the loss function
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    # batch_number = tf.Variable(0, trainable=False)
    # our_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch_number, 1000, 0.8)
    # optimizer = tf.train.AdamOptimizer(our_learning_rate).minimize(cross_entropy, global_step=batch_number)
    # calculate the prediction and the accuracy
    
    raw_prediction = tf.argmax(y_conv, 1)
    raw_prediction_correct = tf.cast(tf.equal(raw_prediction, tf.argmax(y_, 1)), tf.float32)
    raw_accuracy = tf.reduce_mean(raw_prediction_correct)

    max_prob_prediction = tf.argmax(tf.reduce_sum(y_conv, 0), 0)
    max_prob_prediction_correct = tf.cast(tf.equal(max_prob_prediction, tf.argmax(label)), tf.int32)

    vote_count = tf.bincount(tf.cast(raw_prediction, tf.int32))
    maj_vote_prediction = tf.argmax(vote_count)
    maj_vote_prediction_correct = tf.cast(tf.equal(maj_vote_prediction, tf.argmax(label)), tf.int32)

    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    raw_acc_summary = tf.summary.scalar('Raw Accuracy', raw_accuracy)

    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([img_summary, raw_acc_summary])
    training_summary = tf.summary.merge([img_summary, loss_summary])
    test_summary = tf.summary.merge([img_summary, raw_acc_summary])

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
            # (train_samples, train_labels) = gztan.getTrainData()
            # _, summary_str = sess.run([optimizer, training_summary], feed_dict={x: train_samples, train_flag: [True], y_: train_labels})

            for batchNum in range(FLAGS.num_batches):
                (train_samples, train_labels) = gztan.getTrainBatch(batchNum)
                _, summary_str = sess.run([optimizer, training_summary], feed_dict={x: train_samples, train_flag: [True], y_: train_labels})

            if step % (FLAGS.log_frequency + 1) == 0:
                summary_writer.add_summary(summary_str, step)

            # Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                # (test_samples, test_labels) = gztan.getTestData()
                # validation_accuracy, summary_str = sess.run([raw_accuracy, validation_summary], feed_dict={x: test_samples, train_flag: [False], y_: test_labels})

                total_accuracy = 0
                for batchNum in range(FLAGS.num_batches):
                    (test_samples, test_labels) = gztan.getTestBatch(batchNum)
                    validation_accuracy, summary_str = sess.run([raw_accuracy, validation_summary], feed_dict={x: test_samples, train_flag: [False], y_: test_labels})
                    total_accuracy += validation_accuracy

                total_accuracy = total_accuracy / FLAGS.num_batches
                print('step %d, accuracy on validation batch: %g' % (step, total_accuracy))
                summary_writer_validation.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Testing

        # resetting the internal batch indexes
        mp_pred_correct = []
        mv_pred_correct = []

        print('num test tracks: {}'.format(gztan.nTracks))
        for track_id in range(gztan.nTracks):
            (track_samples, track_label) = gztan.getTrackSamples(track_id)
            test_mp_prediction_correct = sess.run(max_prob_prediction_correct, feed_dict={x: track_samples, train_flag: [False], label: track_label})
            test_mv_prediction_correct = sess.run(maj_vote_prediction_correct, feed_dict={x: track_samples, train_flag: [False], label: track_label})
            test_raw_predictions = sess.run(raw_prediction, feed_dict={x: track_samples, train_flag: [False]})
            
            incorrect_pred_idxs = np.where(test_raw_predictions != np.argmax(track_label))[0]

            # print('test_prediction_correct: {}'.format(test_prediction_correct))
            mp_pred_correct.append(test_mp_prediction_correct)
            mv_pred_correct.append(test_mv_prediction_correct)

            if track_id == 0:
                sample = np.array(gztan.getOriginalSample(track_id, incorrect_pred_idxs[0]))
                print('shape of sample: {}'.format(sample.shape))
                librosa.output.write_wav('example.wav', y=sample, sr=22050)

                sample_spec = track_samples[incorrect_pred_idxs[0]]
                # print('shape of sample_spec'.format(sample_spec.shape))
                specshow(sample_spec.reshape([80, 80]), y_axis='linear')

                pylab.savefig('example.png', bbox_inches=None, pad_inches=0)
                pylab.close()

        test_mp_accuracy = sum(mp_pred_correct) / len(mp_pred_correct)
        test_mv_accuracy = sum(mv_pred_correct) / len(mv_pred_correct)
        print('test set: max prob accuracy on test set: %0.3f' % test_mp_accuracy)
        print('test set: maj vote accuracy on test set: %0.3f' % test_mv_accuracy)

if __name__ == '__main__':
    tf.app.run(main=main)
