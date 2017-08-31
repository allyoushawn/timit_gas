#!/usr/bin/env python3
"""
Training RNN autoencoder
"""
import tensorflow as tf
import numpy as np
import random
import sys
from utils import nn_model
from utils.config import TrainConfig, DecodeConfig
from utils.data_parser import load_data_into_mem, padding


#Data config
scp_file = 'feature_scp/train.scp'
dev_scp_file = 'feature_scp/dev.scp'

max_len = 777
min_epoch = 20




def print_progress(progress, total, loss, output_msg):
    sys.stdout.write('\b'*len(output_msg))
    new_output_msg = "Progress: {}/{} loss:{:.4f}"\
     .format(progress, total, loss)
    sys.stdout.write(new_output_msg)
    sys.stdout.flush()
    return new_output_msg




if __name__ == '__main__':

    tr_config = TrainConfig("info")
    tr_config.max_len = max_len
    tr_config.show_config()
    sys.stdout.flush()

    max_len = max_len
    tr_dropout = tr_config.dropout_keep_prob
    tr_zoneout = tr_config.zoneout_keep_prob
    max_epoch = tr_config.max_epoch
    tr_batch = tr_config.batch_size
    print("=============================================================")
    print("                      Loading data                          ")
    print("=============================================================")
    sys.stdout.flush()
    data_list = load_data_into_mem(scp_file)
    dev_data_list = load_data_into_mem(dev_scp_file)
    feature_dim = len(data_list[0][0])
    train_utt_num = len(data_list)
    dev_utt_num = len(dev_data_list)
    print("feature dim: " + str(feature_dim))
    print("utt_num: " + str(train_utt_num))
    print("dev_utt_num: " + str(dev_utt_num))
    tr_config.feature_dim = feature_dim

    print("=============================================================")
    print("                      Set up models                          ")
    print("=============================================================")
    sys.stdout.flush()
    sess = tf.Session()
    #setup nn model's input tensor
    x = tf.placeholder(tf.float32, [None, max_len, feature_dim])
    y_ = tf.placeholder(tf.float32, [None, max_len, feature_dim])
    batch_size = tf.placeholder(tf.int32)
    add_noise = tf.placeholder(tf.bool)
    dropout_keep_prob = tf.placeholder(tf.float32)
    zoneout_keep_prob = tf.placeholder(tf.float32)
    ##pack the input tensors
    input_tensors = {}
    input_tensors["x"] = x
    input_tensors["y_"] = y_
    input_tensors["batch_size"] = batch_size
    input_tensors["add_noise"] = add_noise
    input_tensors["dropout_keep_prob"] = dropout_keep_prob
    input_tensors["zoneout_keep_prob"] = zoneout_keep_prob

    model = nn_model.NeuralNetwork(tr_config, input_tensors)

    model.setup_train()
    model.init_vars(sess)

    output_msg = ''
    best_dev_recon_loss = float('inf')
    print("=============================================================")
    print("                      Start Training                         ")
    print("=============================================================")
    sys.stdout.flush()
    for epoch in range(1, max_epoch + 1):
        print("[ Epoch {} ]".format(epoch))
        print("")
        random.shuffle(data_list)
        counter = 0
        while counter < train_utt_num:
            remain_utt_num = train_utt_num - counter
            batch_size = min(tr_config.batch_size, remain_utt_num)
            X = padding(data_list[counter:counter + batch_size], \
                max_len, feature_dim)
            counter += len(X)
            model.train(sess, X, X, tr_dropout, tr_zoneout, True, len(X))

            recon_loss = model.get_tensor_val('re_loss', sess, X, X, len(X))

            output_msg = print_progress(counter, train_utt_num, recon_loss,\
                            output_msg)
        print('')
        if epoch >= min_epoch:
            #dev eval
            output_msg = ''
            counter = 0

            ##precision/recall_set dimension: [th * utt]
            recon_loss_list = []
            while counter < dev_utt_num:
                remain_utt_num = dev_utt_num - counter
                batch_size = min(tr_config.batch_size, remain_utt_num)
                X = padding(dev_data_list[counter:counter + batch_size], \
                             max_len, feature_dim)
                counter += len(X)

                recon_loss = model.get_tensor_val('re_loss', sess, X, X, len(X))
                recon_loss_list.append(recon_loss)
                output_msg = print_progress(counter, dev_utt_num, recon_loss,\
                                output_msg)


            dev_recon_loss = sum(recon_loss_list) / len(recon_loss_list)
            print('')
            print('The dev loss is {:.4f}'.format(dev_recon_loss))
            if epoch == min_epoch:
                model.save_vars(sess, tr_config.model_loc)
                best_dev_recon_loss = dev_recon_loss
                continue

            if dev_recon_loss < best_dev_recon_loss:
                model.save_vars(sess, tr_config.model_loc)
                best_dev_recon_loss = dev_recon_loss

            else:
                print('Performance get worse, stop training')
                break

    print('=============================================================')
    print('      Training finished, show config info. again             ')
    print('=============================================================')
    tr_config.show_config()
