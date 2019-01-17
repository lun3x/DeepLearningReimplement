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

tf.app.flags.DEFINE_string('decay', 'const',
                            'Whether to use constant or exp decay learning rate. (default: %(default)s)')
tf.app.flags.DEFINE_string('repr-func', 'mel',
                            'Whether to use mel, cqt_hz, or cqt_note. (default: %(default)s)')
tf.app.flags.DEFINE_string('net-depth', 'shallow',
                            'Whether to use the deep or shallow network. (default: %(default)s)')
tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 100,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 9,
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
tf.app.flags.DEFINE_string('log-dir', '{cwd}/{d}_{rep}_{decay}_{lr}_{steps}/logs/'.format(cwd=os.getcwd(), d=FLAGS.net_depth, rep=FLAGS.repr_func, decay=FLAGS.decay, steps=str(FLAGS.max_steps), lr=str(FLAGS.learning_rate)),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')

# the initialiser object implementing Xavier initialisation
# we will generate weights from the uniform distribution
xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

def shallownn(x, train_flag):
    """shallownn builds the graph for a shallow net for classifying GZTAN audio samples.

  Args:
      x: an input tensor with the dimensions (N_examples, 80, 80, 1), where 80x80 is the
        number of pixels in a spectrogram of a GZTAN audio sample.
      train_flag: a length 1 tensor containing a boolean flag, True if training, False if testing

  Returns:
      y: is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the audio spectrograms into one of 10 classes
        (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
    """

    # First convolutional layer - maps one image to 16 feature maps.
    with tf.variable_scope('Conv_Spectral_1'):
        conv_spec_1 = tf.layers.conv2d(
            inputs=x,
            filters=16,
            kernel_size=[10, 23],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_spec_1'
        )

        h_conv_spec_1 = tf.nn.relu(conv_spec_1, name='conv_spec_1_relu')

        pool_spec_1 = tf.layers.max_pooling2d(
            inputs=h_conv_spec_1,
            pool_size=[1, 20],
            strides=[1, 20],
            name='pool_spec_1'
        )

        reshaped_pool_spec_1 = tf.reshape(pool_spec_1, [-1, 5120])

    with tf.variable_scope('Conv_Temporal_1'):
        conv_temp_1 = tf.layers.conv2d(
            inputs=x,
            filters=16,
            kernel_size=[21, 20],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_spec_1'
        )

        h_conv_temp_1 = tf.nn.relu(conv_temp_1, name='conv_temp_1_relu')

        pool_temp_1 = tf.layers.max_pooling2d(
            inputs=h_conv_temp_1,
            pool_size=[20, 1],
            strides=[20, 1],
            name='pool_temp_1'
        )

        reshaped_pool_temp_1 = tf.reshape(pool_temp_1, [-1, 5120])

    with tf.variable_scope('Merge'):
        merged_streams = tf.concat([reshaped_pool_spec_1, reshaped_pool_temp_1], 1, name='merged_streams')

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
        return fc3

def deepnn(x, train_flag):
    """shallownn builds the graph for a shallow net for classifying GZTAN audio samples.

  Args:
      x: an input tensor with the dimensions (N_examples, 80, 80, 1), where 80x80 is the
        number of pixels in a spectrogram of a GZTAN audio sample.
      train_flag: a length 1 tensor containing a boolean flag, True if training, False if testing

  Returns:
      y: is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the audio spectrograms into one of 10 classes
        (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
    """

    # First convolutional layer - maps one image to 16 feature maps.
    with tf.variable_scope('Conv_Spectral_1'):
        conv_spec_1 = tf.layers.conv2d(
            inputs=x,
            filters=16,
            kernel_size=[10, 23],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_spec_1'
        )

        h_conv_spec_1 = tf.nn.relu(conv_spec_1, name='conv_spec_1_relu')

        # Pooling layer - downsamples by 2X.
        pool_spec_1 = tf.layers.max_pooling2d(
            inputs=h_conv_spec_1,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_spec_1'
        )

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

        h_conv_spec_2 = tf.nn.relu(conv_spec_2, name='conv_spec_2_relu')

        # Pooling layer - downsamples by 2X.
        pool_spec_2 = tf.layers.max_pooling2d(
            inputs=h_conv_spec_2,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_spec_2'
        )

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

        h_conv_spec_3 = tf.nn.relu(conv_spec_3, name='conv_spec_3_relu')

        # Pooling layer - downsamples by 2X.
        pool_spec_3 = tf.layers.max_pooling2d(
            inputs=h_conv_spec_3,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_spec_3'
        )

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

        h_conv_spec_4 = tf.nn.relu(conv_spec_4, name='conv_spec_4_relu')

        pool_spec_4 = tf.layers.max_pooling2d(
            inputs=h_conv_spec_4,
            pool_size=[1, 5],
            strides=[1, 5],
            name='pool_spec_4'
        )

        reshaped_pool_spec_4 = tf.reshape(pool_spec_4, [-1, 2560])


    with tf.variable_scope('Conv_Temporal_1'):
        conv_temp_1 = tf.layers.conv2d(
            inputs=x,
            filters=16,
            kernel_size=[21, 10],
            padding='same',
            use_bias=False,
            kernel_initializer=xavier_initializer,
            name='conv_temp_1'
        )

        h_conv_temp_1 = tf.nn.relu(conv_temp_1, name='conv_temp_1_relu')

        # Pooling layer - downsamples by 2X.
        pool_temp_1 = tf.layers.max_pooling2d(
            inputs=h_conv_temp_1,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_temp_1'
        )

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

        h_conv_temp_2 = tf.nn.relu(conv_temp_2, name='conv_temp_2_relu')

        # Pooling layer - downsamples by 2X.
        pool_temp_2 = tf.layers.max_pooling2d(
            inputs=h_conv_temp_2,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_temp_2'
        )

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

        h_conv_temp_3 = tf.nn.relu(conv_temp_3, name='conv_temp_3_relu')

        # Pooling layer - downsamples by 2X.
        pool_temp_3 = tf.layers.max_pooling2d(
            inputs=h_conv_temp_3,
            pool_size=[2, 2],
            strides=[2, 2],
            name='pool_temp_3'
        )

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

        h_conv_temp_4 = tf.nn.relu(conv_temp_4, name='conv_temp_4_relu')

        pool_temp_4 = tf.layers.max_pooling2d(
            inputs=h_conv_temp_4,
            pool_size=[5, 1],
            strides=[5, 1],
            name='pool_temp_4'
        )

        reshaped_pool_temp_4 = tf.reshape(pool_temp_4, [-1, 2560])

    with tf.variable_scope('Merge'):
        merged_streams = tf.concat([reshaped_pool_spec_4, reshaped_pool_temp_4], 1, name='merged_streams')

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
        return fc3

def main(_):
    tf.reset_default_graph()

    # Import data
    gztan = GZTan(FLAGS.num_batches, mel=(FLAGS.repr_func == 'mel'))

    # print('num train tracks: {}'.format(gztan.nTrainTracks))

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
        y_conv = shallownn(x, train_flag)
    elif FLAGS.net_depth == 'deep':
        print('DEEP')
        y_conv = deepnn(x, train_flag)
    else:
        print("Error: Unrecognised depth.")
        return

    # Define loss function - softmax_cross_entropy + L1 regularisation
    with tf.name_scope("regularized_loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0001)
        weights = tf.trainable_variables()
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        regularized_cross_entropy = tf.add(cross_entropy, regularization_penalty, name='reg_loss')

    # Define AdamOptimiser, using FLAGS.learning_rate to minimize the loss function
    if FLAGS.decay == 'const':
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(regularized_cross_entropy)
    else:
        batch_number = tf.Variable(0, trainable=False)
        our_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch_number, 3000, 0.9)
        optimizer = tf.train.AdamOptimizer(our_learning_rate).minimize(regularized_cross_entropy, global_step=batch_number)
        
    # Calculate the prediction and the accuracy
    raw_prediction = tf.argmax(y_conv, 1)
    raw_prediction_correct = tf.cast(tf.equal(raw_prediction, tf.argmax(y_, 1)), tf.float32)
    raw_accuracy = tf.reduce_mean(raw_prediction_correct)

    max_prob_prediction = tf.argmax(tf.reduce_sum(y_conv, 0), 0)
    max_prob_prediction_correct = tf.cast(tf.equal(max_prob_prediction, tf.argmax(label)), tf.int32)

    vote_count = tf.bincount(tf.cast(raw_prediction, tf.int32))
    maj_vote_prediction = tf.argmax(vote_count)
    maj_vote_prediction_correct = tf.cast(tf.equal(maj_vote_prediction, tf.argmax(label)), tf.int32)

    av_confidence = tf.reduce_mean(y_conv, 0)

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(FLAGS.log_dir + '/_validate', sess.graph, flush_secs=5)

        sess.run(tf.global_variables_initializer())

        # Training and validation
        for step in range(FLAGS.max_steps):
            # Training: Backpropagation using train set
            total_loss = 0
            for batchNum in range(FLAGS.num_batches):
                (train_samples, train_labels) = gztan.getTrainBatch(batchNum)
                _, batch_loss = sess.run([optimizer, regularized_cross_entropy], feed_dict={x: train_samples, train_flag: [True], y_: train_labels})
                total_loss += batch_loss

            if step % (FLAGS.log_frequency + 1) == 0:
                loss_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="Regularized_Loss", simple_value=total_loss), 
                ])
                summary_writer.add_summary(loss_summary, step)

            # Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                total_accuracy = 0.0
                for batchNum in range(FLAGS.num_batches):
                    (test_samples, test_labels) = gztan.getTestBatch(batchNum)
                    validation_accuracy = sess.run(raw_accuracy, feed_dict={x: test_samples, train_flag: [False], y_: test_labels})
                    total_accuracy += validation_accuracy

                total_accuracy = total_accuracy / FLAGS.num_batches
                print('step %d, accuracy on validation batch: %g' % (step, total_accuracy))

                tot_acc_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="Total_Raw_Accuracy", simple_value=total_accuracy), 
                ])
                summary_writer_validation.add_summary(tot_acc_summary, step)

            # # Save the model checkpoint periodically.
            # if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
            #     checkpoint_path = FLAGS.log_dir + '/_train' + '/model.ckpt'
            #     saver.save(sess, checkpoint_path, global_step=step)

            gztan.shuffle()

        # Testing
        mp_pred_correct = []
        mv_pred_correct = []
        raw_pred_acc = []
        done = False

        print('num test tracks: {}'.format(gztan.nTracks))
        confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int32)
        for track_id in range(gztan.nTracks):
            (track_samples, track_labels) = gztan.getTrackSamples(track_id)
            track_label = track_labels[0]
            test_raw_acc = sess.run(raw_accuracy, feed_dict={x: track_samples, train_flag: [False], y_: track_labels})

            test_mp_prediction = sess.run(max_prob_prediction, feed_dict={x: track_samples, train_flag: [False]})
            test_mp_prediction_correct = (test_mp_prediction == np.argmax(track_label))
            test_mv_prediction = sess.run(maj_vote_prediction, feed_dict={x: track_samples, train_flag: [False], label: track_label})
            test_mv_prediction_correct = (test_mv_prediction == np.argmax(track_label))
            
            confusion_matrix[int(np.argmax(track_label)), int(test_mp_prediction)] += 1

            mp_pred_correct.append(test_mp_prediction_correct)
            mv_pred_correct.append(test_mv_prediction_correct)
            raw_pred_acc.append(test_raw_acc)

            # Find interesting examples and output them
            if not test_mv_prediction_correct and not test_mp_prediction_correct and not done:
                test_raw_confidences = sess.run(y_conv, feed_dict={x: track_samples, train_flag: [False]})
                test_raw_predictions = np.argmax(test_raw_confidences, axis=1)

                test_av_conf_vals = np.mean(test_raw_confidences, axis=1)
                low_correct_confidences = np.where(test_raw_confidences[:,np.argmax(track_label)] < test_av_conf_vals)[0]

                if len(low_correct_confidences) > 0:
                    done = True
                    
                # np.where outputs a 1-tuple so do [0] on this to get actual result
                print('test_mp_prediction: {} test_mv_prediction: {} true label: {}'.format(test_mp_prediction, test_mv_prediction, np.argmax(track_label)))
                incorrect_pred_idxs = np.where(test_raw_predictions != np.argmax(track_label))[0]
                print('found at track_id: {}!'.format(track_id))
                for idx in low_correct_confidences:
                    print('Incorrectly classified sample {} with as {} with confidences {}. Should be {}.'.format(idx, test_raw_predictions[idx], test_raw_confidences[idx], np.argmax(track_label)))
                    gztan.outputSample(track_id, idx)
                    
                    sample_spec = track_samples[idx]
                    specshow(sample_spec.reshape([80, 80]), y_axis=FLAGS.repr_func)

                    pylab.savefig('incorrect_{r}_track{t}_example{e}.png'.format(r=FLAGS.repr_func, t=track_id, e=idx), bbox_inches=None, pad_inches=0)
                    pylab.close()
        
        test_mp_accuracy = sum(mp_pred_correct) / len(mp_pred_correct)
        test_mv_accuracy = sum(mv_pred_correct) / len(mv_pred_correct)
        test_raw_accuracy = sum(raw_pred_acc) / len(raw_pred_acc)
        print('test set: raw accuracy on test set: %0.3f' % test_raw_accuracy)
        print('test set: max prob accuracy on test set: %0.3f' % test_mp_accuracy)
        print('test set: maj vote accuracy on test set: %0.3f' % test_mv_accuracy)

        np.savetxt("confusion.csv", confusion_matrix, delimiter=",")

if __name__ == '__main__':
    tf.app.run(main=main)
