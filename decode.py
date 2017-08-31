#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import sys
from utils import nn_model
from utils.config import TrainConfig, DecodeConfig
from utils.data_parser import bound_loader, load_data_into_mem, padding
from utils.data_parser import get_specific_phn_bound
from utils.eval import r_val_eval, thresh_segmentation_eval
import pdb
import matplotlib.pyplot as plt

#Data config
test_scp_file = 'feature_scp/test.scp'
test_bound_file = 'bounds/phn/test.phn'
max_len = 777

enable_plt = False




def print_progress(progress, total, precision, recall, output_msg):
    output_msg = 'Progress: {}/{} Precision:{:.4f}, Recall:{:.4f}'.format(progress, total, precision, recall)
    sys.stdout.write('\b'*len(output_msg))
    sys.stdout.write(output_msg)
    sys.stdout.flush()




if __name__ == '__main__':
    de_config = DecodeConfig('decode_info')
    tr_config = TrainConfig('info')
    tr_config.max_len = max_len
    tr_config.show_config()
    de_config.show_config()

    nn_hidden_num = tr_config.nn_hidden_num
    rnn_cell_num = tr_config.rnn_cell_num
    max_len = max_len
    init_lr = tr_config.learning_rate
    noise_prob = tr_config.noise_magnitude
    tr_dropout = tr_config.dropout_keep_prob
    tr_zoneout = tr_config.zoneout_keep_prob
    max_epoch = tr_config.max_epoch
    tr_batch = tr_config.batch_size
    tolerance_window = de_config.tolerance_window
    print('=============================================================')
    print('                      Loading data                          ')
    print('=============================================================')
    sys.stdout.flush()
    data_list = load_data_into_mem(test_scp_file)
    feature_dim = len(data_list[0][0])
    total_test_utt_num = len(data_list)
    print('feature dim: ' + str(feature_dim))
    print('utt_num: ' + str(total_test_utt_num))
    tr_config.feature_dim = feature_dim

    print('=============================================================')
    print('                      Set up models.                         ')
    print('=============================================================')
    #setup nn model's input tensor
    x = tf.placeholder(tf.float32, [None, max_len, feature_dim])
    y_ = tf.placeholder(tf.float32, [None, max_len, feature_dim])
    batch_size = tf.placeholder(tf.int32)
    add_noise = tf.placeholder(tf.bool)
    dropout_keep_prob = tf.placeholder(tf.float32)
    zoneout_keep_prob = tf.placeholder(tf.float32)
    ##pack the input tensors
    input_tensors = {}
    input_tensors['x'] = x
    input_tensors['y_'] = y_
    input_tensors['batch_size'] = batch_size
    input_tensors['add_noise'] = add_noise
    input_tensors['dropout_keep_prob'] = dropout_keep_prob
    input_tensors['zoneout_keep_prob'] = zoneout_keep_prob

    model = nn_model.NeuralNetwork(tr_config, input_tensors)
    sess = tf.Session()
    model.restore_vars(sess, de_config.model_loc)

    if enable_plt == True:
        bounds = get_specific_phn_bound('/media/hdd/csie/corpus/timit/test/dr4/fadg0/sa1.phn')
        for bound in bounds:
            plt.axvline(x=bound)

    bound_generator = bound_loader(test_bound_file, de_config.batch_size)
    print('')
    output_msg = ''
    recall_list = []
    precision_list = []
    '''
    #The th for LSTM
    th = [0.05, 0.03, 0.028, 0.026, 0.024, 0.022, 0.02, 0.018]
    th += [ 0.016, 0.014, 0.012, 0.01, 0.008, 0.004, 0.002, 0.001]
    '''
    #The th for GRU
    th = [0.05, 0.03, 0.01, 0.009, 0.008, 0.0075, 0.007, 0.0065]
    th += [0.006, 0.0055, 0.005, 0.0045, 0.004, 0.003, 0.002, 0.001]


    for i in range(len(th)):
        recall_list.append([])
        precision_list.append([])

    #precision/recall_set dimension: [th * utt]

    counter = 0
    print('=============================================================')
    print('                      Start Decoding                         ')
    print('=============================================================')
    while counter < total_test_utt_num:
        remain_utt_num = total_test_utt_num - counter
        batch_size = min(de_config.batch_size, remain_utt_num)
        X = padding(data_list[counter:counter + batch_size], \
            max_len, feature_dim)
        bounds_list = next(bound_generator)
        counter += len(X)

        gas = model.get_tensor_val('encoder_g1', sess, X, X, len(X))
        gas = np.mean(gas, axis=2)

        for th_idx, t_f in enumerate(th):
            batch_recall_list, batch_precision_list = \
                thresh_segmentation_eval(gas[:,1:] - gas[:,:-1], bounds_list,\
                                         de_config.tolerance_window, t_f)
            if th_idx == 5:
                print_progress(counter, total_test_utt_num, \
                                batch_precision_list[0], \
                                batch_recall_list[0], output_msg)

            precision_list[th_idx] += batch_precision_list
            recall_list[th_idx] += batch_recall_list

        if enable_plt == True:
            T = range(len(gas[0]))
            plt.plot(T, gas[0], color='red', linewidth=4)
            pdb.set_trace()

    print('')
    title =  'thresh precision recall f_score r_val'
    print(title)

    r_val_list = []
    for t_idx, t in enumerate(th):
        precision = sum(precision_list[t_idx]) / len(precision_list[t_idx])
        recall = sum(recall_list[t_idx]) / len(recall_list[t_idx])
        recall *= 100
        precision *= 100
        if recall == 0. or precision == 0.:
            f_score = -1.
            r_val = -1.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
            r_val = r_val_eval(precision, recall)
        r_val_list.append(r_val)

        print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'. \
            format(t, precision, recall, f_score, r_val))
    print('')
    print('The best r_val is: {:.4f}'.format(max(r_val_list)))
