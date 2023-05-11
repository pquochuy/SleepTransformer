import os
import numpy as np
import tensorflow as tf


import shutil, sys
from datetime import datetime
import h5py
import hdf5storage

from sleeptransformer import SleepTransformer
from config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_wrapper import DataGeneratorWrapper

from scipy.io import loadmat, savemat

import time

from tensorflow.python import pywrap_tensorflow

import time

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

#tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("seq_len", 20, "Sequence length (default: 10)")

tf.app.flags.DEFINE_integer("num_blocks", 0, "Number of transformer block (default: 0)") # if zero, specific parameters are expected for the numbers of frame blocks and seq blocks
tf.app.flags.DEFINE_integer("frm_num_blocks", 1, "Number of transformer block (default: 0)")
tf.app.flags.DEFINE_integer("seq_num_blocks", 1, "Number of transformer block (default: 0)")
tf.app.flags.DEFINE_float("frm_fc_dropout", 0.1, "Dropout keep probability (default: 0.1)")
tf.app.flags.DEFINE_float("frm_attention_dropout", 0.1, "Dropout keep probability (default: 0.1)")
tf.app.flags.DEFINE_float("seq_fc_dropout", 0.1, "Dropout keep probability (default: 0.1)")
tf.app.flags.DEFINE_float("seq_attention_dropout", 0.1, "Dropout keep probability (default: 0.1)")
tf.app.flags.DEFINE_float("fc_dropout", 0.1, "Dropout keep probability (default: 0.1)")


FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

config = Config()
config.epoch_seq_len = FLAGS.seq_len
config.epoch_step = FLAGS.seq_len

if (FLAGS.num_blocks > 0):
    config.frm_num_blocks = FLAGS.num_blocks
    config.seq_num_blocks = FLAGS.num_blocks
else:
    config.frm_num_blocks = FLAGS.frm_num_blocks
    config.seq_num_blocks = FLAGS.seq_num_blocks

config.frm_fc_dropout = FLAGS.frm_fc_dropout  # 0.3 dropout
config.frm_attention_dropout = FLAGS.frm_fc_dropout  # 0.3 dropout
config.seq_fc_dropout = FLAGS.frm_fc_dropout  # 0.3 dropout
config.seq_attention_dropout = FLAGS.frm_fc_dropout  # 0.3 dropout
config.fc_dropout = FLAGS.frm_fc_dropout


eeg_active = ((FLAGS.eeg_train_data != "") and (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") and (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") and (FLAGS.emg_test_data != ""))

if (not eog_active and not emg_active):
    print("eeg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             num_fold=config.num_fold_training_data,
                                             #data_shape_1=[config.ntime],
                                             data_shape_2=[config.frame_seq_len, config.ndim],  # excluding 0th element
                                             seq_len = config.epoch_seq_len,
                                             shuffle=True)
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                             num_fold=config.num_fold_testing_data,
                                             #data_shape_1=[config.ntime],
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    nchannel = 1

elif(eog_active and not emg_active):
    print("eeg and eog active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_train_data),
                                             num_fold=config.num_fold_training_data,
                                             #data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=True)
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                                  eog_filelist=os.path.abspath(FLAGS.eog_test_data),
                                             num_fold=config.num_fold_testing_data,
                                             #data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_wrapper.compute_eog_normalization_params()
    test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    test_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    nchannel = 2
elif(eog_active and emg_active):
    print("eeg, eog, and emg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_train_data),
                                             emg_filelist=os.path.abspath(FLAGS.emg_train_data),
                                             num_fold=config.num_fold_training_data,
                                             #data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=True)
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                                  eog_filelist=os.path.abspath(FLAGS.eog_test_data),
                                                  emg_filelist=os.path.abspath(FLAGS.emg_test_data),
                                             num_fold=config.num_fold_testing_data,
                                             #data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_wrapper.compute_eog_normalization_params()
    train_gen_wrapper.compute_emg_normalization_params()
    test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    test_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    test_gen_wrapper.set_emg_normalization_params(train_gen_wrapper.emg_meanX, train_gen_wrapper.emg_stdX)
    nchannel = 3

config.nchannel = nchannel
config.seq_d_model = config.ndim*config.nchannel
config.frm_d_model = config.ndim*config.nchannel

# do not need training data anymore
del train_gen_wrapper

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.475, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = SleepTransformer(config=config)

        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        best_dir = os.path.join(checkpoint_path, "best_model_acc")


        variables = list()
        variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        #restorer = tf.train.Saver(variables)
        print("RESTORE VARIABLES")
        #print(variables)
        for i, v in enumerate(variables):
            print(v.name[:-2])

        vars_in_checkpoint = tf.train.list_variables(best_dir)
        print("IN-CHECK-POINT VARIABLES")
        #print(vars_in_checkpoint)
        vars_in_checkpoint_names = list()
        for i, v in enumerate(vars_in_checkpoint):
            print(v[0])
            vars_in_checkpoint_names.append(v[0])

        var_list_to_retstore = [v for v in variables if v.name[:-2] in vars_in_checkpoint_names]
        print("ACTUAL RESTORE VARIABLES")
        print(var_list_to_retstore)


        restorer = tf.train.Saver(var_list_to_retstore)
        #restorer = tf.train.Saver(tf.all_variables())
        # Load pretrained model
        restorer.restore(sess, best_dir)
        print("Model loaded")


        def dev_step(x_batch, y_batch):
            '''
            A single evaluation step
            '''
            feed_dict = {
                net.input_x: x_batch,
                net.input_y: y_batch,
                net.istraining: 0
            }
            output_loss, total_loss, yhat, score = sess.run(
                   [net.output_loss, net.loss, net.predictions, net.scores], feed_dict)
            return output_loss, total_loss, yhat, score

        def evaluate(gen_wrapper):

            N = int(np.sum(gen_wrapper.file_sizes) - (config.epoch_seq_len - 1)*len(gen_wrapper.file_sizes))
            yhat = np.zeros([N, config.epoch_seq_len])
            y = np.zeros([N, config.epoch_seq_len])

            score = np.zeros([N, config.epoch_seq_len, config.nclass])

            count = 0
            output_loss = 0
            total_loss = 0
            gen_wrapper.new_subject_partition()
            for data_fold in range(config.num_fold_testing_data):
                # load data of the current fold
                gen_wrapper.next_fold()
                yhat_, score_, output_loss_, total_loss_ = _evaluate(gen_wrapper.gen)

                output_loss += output_loss_
                total_loss += total_loss_

                yhat[count : count + len(gen_wrapper.gen.data_index)] = yhat_
                score[count : count + len(gen_wrapper.gen.data_index)] = score_

                # groundtruth
                for n in range(config.epoch_seq_len):
                    y[count : count + len(gen_wrapper.gen.data_index), n] =\
                        gen_wrapper.gen.label[gen_wrapper.gen.data_index - (config.epoch_seq_len - 1) + n]
                count += len(gen_wrapper.gen.data_index)

            test_acc = np.zeros([config.epoch_seq_len])
            for n in range(config.epoch_seq_len):
                test_acc[n] = accuracy_score(yhat[:,n], y[:,n]) # excluding the indexes of the recordings

            return test_acc, yhat, score, output_loss, total_loss

        def _evaluate(gen):
            # Validate the model on the entire data in gen
            output_loss =0
            total_loss = 0

            factor = 10
            yhat = np.zeros([len(gen.data_index), config.epoch_seq_len])
            score = np.zeros([len(gen.data_index), config.epoch_seq_len, config.nclass])
            # use 10x of minibatch size to speed up
            num_batch_per_epoch = np.floor(len(gen.data_index) / (factor*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, label_batch_ = gen.next_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                yhat[(test_step - 1) * factor * config.batch_size: test_step * factor * config.batch_size] = yhat_
                score[(test_step - 1) * factor * config.batch_size: test_step * factor * config.batch_size] = score_
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x_batch, y_batch, label_batch_ = gen.rest_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, y_batch)
                yhat[(test_step - 1) * factor * config.batch_size: len(gen.data_index)] = yhat_
                score[(test_step - 1) * factor * config.batch_size: len(gen.data_index)] = score_
                output_loss += output_loss_
                total_loss += total_loss_
            yhat = yhat + 1 # make label starting from 1 rather than 0

            return yhat, score, output_loss, total_loss

        # evaluation on test data
        start_time = time.time()
        test_acc, test_yhat, test_score, test_output_loss, test_total_loss = evaluate(gen_wrapper=test_gen_wrapper)
        end_time = time.time()
        with open(os.path.join(out_dir, "test_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))
        
        hdf5storage.savemat(os.path.join(out_path, "test_ret.mat"),
                {'yhat': test_yhat, 
                    'acc': test_acc, 
                    'score': test_score, 
                    'output_loss': test_output_loss, 
                    'total_loss': test_total_loss}, 
                format='7.3')

        test_gen_wrapper.gen.reset_pointer()
