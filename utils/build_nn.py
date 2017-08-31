#!/usr/bin/env python2
import tensorflow as tf
from  tensorflow.python.ops import nn_ops
import local_rnn.local_rnn_cell




def build_dnn(x, input_dim, output_dim):
    w = tf.Variable(tf.truncated_normal([input_dim, output_dim],
                                        stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[output_dim]))
    return tf.nn.relu(tf.matmul(x, w) + b)


def zoneout(x_candidate, x, keep_prob):
    new_x = keep_prob * nn_ops.dropout(x_candidate - x, keep_prob) + x

    return new_x


def build_rnn(rnn_cell_num, num_steps, rnn_input, batch_size, y_, \
               dropout_keep_prob, zoneout_keep_prob, var_scope):
    rnn_cell = local_rnn.local_rnn_cell.GRUCell(rnn_cell_num)
    if var_scope != "seg_rnn":
        rnn_cell = local_rnn.local_rnn_cell.DropoutWrapper(rnn_cell, \
                    input_keep_prob=dropout_keep_prob, \
                    output_keep_prob=dropout_keep_prob)

    state = rnn_cell.zero_state(batch_size, tf.float32)
    outputs = []
    states = []
    reset_gate_val = []
    update_gate_val = []
    mask_sequence = tf.sign(tf.reduce_max(tf.abs(y_), reduction_indices=2))
    with tf.variable_scope(var_scope):
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()

            output, state, r, u = rnn_cell(rnn_input[:, time_step, :], state)
            #Zoneout
            if time_step > 0:
                state = zoneout(state, states[time_step - 1], zoneout_keep_prob)
                state.set_shape((None, rnn_cell_num))

            mask_t = tf.expand_dims(mask_sequence[:, time_step], 1)
            mask_t = tf.tile(mask_t, [1, rnn_cell_num])
            output = output * mask_t
            r = r * mask_t
            u = u * mask_t
            state = state * mask_t

            outputs.append(output)
            states.append(state)
            reset_gate_val.append(r)
            update_gate_val.append(u)

    rnn_output = tf.reshape(tf.concat(1, outputs),
                            (batch_size, -1, rnn_cell_num))
    rnn_output.set_shape((None, None, rnn_cell_num))
    reset_gate_val = tf.reshape(tf.concat(1, reset_gate_val),\
                          (batch_size, -1, rnn_cell_num))
    update_gate_val = tf.reshape(tf.concat(1, update_gate_val),\
                          (batch_size, -1, rnn_cell_num))

    return rnn_output, reset_gate_val, update_gate_val


def gate_diff(g_val):
    g_post = tf.slice(g_val, [0, 1], [-1, -1])
    g_post = tf.pad(g_post, [[0, 0], [0, 1]], "CONSTANT")
    g_diff = tf.sub(g_post, g_val)

    return g_diff


def seg_rnn_label_from_thresh(g_diff, tr_config):
    g_diff_mean = tf.reduce_mean(g_diff, reduction_indices=1)
    th_mat = tf.expand_dims(g_diff_mean, 1)
    th_mat = tr_config.seg_rnn_th * tf.tile(th_mat, [1, tr_config.num_steps])
    label = tf.ceil(tf.sub(g_diff, th_mat))
    return tf.one_hot(tf.cast(label, tf.int32), 2)


def segment_rnn_loss(seg_rnn_output, seg_label, mask, tr_config):
    soft_max_input = tf.reshape(seg_rnn_output, [-1, tr_config.rnn_cell_num])
    W_seg = tf.Variable(tf.truncated_normal([rnn_cell_num, 2], stddev=0.1))
    b_seg = tf.Variable(tf.constant(0.1, shape=[2]))
    logits = tf.nn.relu(tf.matmul(soft_max_input, W_seg) + b_seg)
    seg_predict = tf.nn.softmax(logits)
    seg_predict = tf.reshape(seg_predict, [-1, num_steps, 2])

    xent = tf.reduce_sum(-seg_label * tf.log(seg_predict), reduction_indices=2)
    xent *= mask
    xent = tf.reduce_sum(xent, reduction_indices=1)
    xent /= tf.reduce_sum(mask, reduction_indices=1)

    seg_mask = tf.reduce_max(seg_label, reduction_indices=2)
    seg_mask = tf.reduce_max(seg_mask, reduction_indices=1)

    return  tf.reduce_sum(xent * seg_mask) / tf.reduce_sum(seg_mask)


def reconstruction_loss(y, y_, mask):
    square_error = tf.mul(tf.sub(y_, y), tf.sub(y_, y))
    square_error = tf.reduce_mean(square_error, reduction_indices=2)
    square_error *= mask

    sum_square_error = tf.reduce_sum(square_error, reduction_indices=1)
    sum_square_error /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(sum_square_error)

def build_whole_nn_model(x, y_, batch_size, add_noise, dropout_keep_prob, zoneout_keep_prob, tr_config):
    nn_hidden_layer_num = tr_config.nn_hidden_layer_num
    feature_dim = tr_config.feature_dim
    nn_hidden_num = tr_config.nn_hidden_num
    rnn_cell_num = tr_config.rnn_cell_num
    num_steps = tr_config.num_steps
    init_lr = tr_config.learning_rate
    noise_prob = tr_config.noise_magnitude
    tr_dropout = tr_config.dropout_keep_prob
    tr_zoneout = tr_config.zoneout_keep_prob
    max_epoch = tr_config.epoch_num
    tr_batch = tr_config.batch_size
    tolerance_window = tr_config.tolerance_window
    rnd = tf.random_uniform(tf.shape(x))
    noise_th = tr_config.noise_magnitude * tf.ones_like(x, tf.float32)
    noise_mask = tf.ceil(tf.sub(rnd, noise_th))

    model_input = tf.cond(add_noise, lambda: tf.mul(x, noise_mask), lambda: x)
    with tf.variable_scope('autoencoder_model'):
        with tf.variable_scope('encoder'):
            dnn_input = tf.reshape(model_input, [-1, feature_dim])
            dnn_output = build_dnn(dnn_input, feature_dim, nn_hidden_num)
            rnn_input = tf.reshape(dnn_output, [-1, num_steps, nn_hidden_num])

            rnn_code, reset_gate_val, \
             update_gate_val = build_rnn(rnn_cell_num, num_steps, rnn_input, \
                                          batch_size, \
                                          y_, dropout_keep_prob, zoneout_keep_prob, "encoder_rnn")

            reset_gate_val = tf.reduce_mean(reset_gate_val, reduction_indices=2)
            update_gate_val = tf.reduce_mean(update_gate_val, reduction_indices=2)

        with tf.variable_scope('decoder'):
            rnn_output, _, _ = build_rnn(rnn_cell_num, num_steps, \
                                          rnn_code, batch_size, y_, \
                                          dropout_keep_prob, zoneout_keep_prob, "decoder_rnn")

            dnn_input = tf.reshape(rnn_output, [-1, rnn_cell_num])
            dnn_output = build_dnn(dnn_input, rnn_cell_num, nn_hidden_num)

            W_output = tf.Variable(tf.truncated_normal([nn_hidden_num, feature_dim], stddev=0.1))
            b_output = tf.Variable(tf.constant(0.1, shape=[feature_dim]))
            decoder_output = tf.matmul(dnn_output, W_output) + b_output

        y = tf.reshape(decoder_output, [-1, num_steps, feature_dim])

    mask = tf.sign(tf.reduce_max(tf.abs(y_), reduction_indices=2))
    sequence_len = tf.reduce_sum(mask, reduction_indices=1)

    re_loss = reconstruction_loss(y, y_, mask)
    return re_loss, y, reset_gate_val, update_gate_val
