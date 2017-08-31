#!/usr/bin/env python3
import tensorflow as tf
from  tensorflow.python.ops import nn_ops
from  local_rnn import local_cell
from local_rnn import local_rnn
from IPython.core.debugger import Tracer



def build_dnn(x, input_dim, output_dim):
    w = tf.Variable(tf.truncated_normal([input_dim, output_dim],
                                        stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[output_dim]))
    return tf.nn.relu(tf.matmul(x, w) + b)


def zoneout(x_candidate, x, keep_prob):
    new_x = keep_prob * nn_ops.dropout(x_candidate - x, keep_prob) + x

    return new_x


def build_hand_crafted_rnn(
               rnn_cell_num, max_len, rnn_input, batch_size, rnn_mask, \
               dropout_keep_prob, zoneout_keep_prob, var_scope):
    rnn_cell = local_gru_cell.GRUCell(rnn_cell_num)
    rnn_cell = local_gru_cell.DropoutWrapper(rnn_cell, \
                input_keep_prob=dropout_keep_prob, \
                output_keep_prob=dropout_keep_prob)

    state = rnn_cell.zero_state(batch_size, tf.float32)
    outputs = []
    reset_gate_val = []
    update_gate_val = []
    with tf.variable_scope(var_scope):
        for time_step in range(max_len):
            if time_step > 0: tf.get_variable_scope().reuse_variables()

            output, state, r, u = rnn_cell(rnn_input[:, time_step, :], state)
            outputs.append(output)
            reset_gate_val.append(r)
            update_gate_val.append(u)

    rnn_output = tf.reshape(tf.concat(outputs, 1), \
                            (batch_size, -1, rnn_cell_num))
    rnn_output.set_shape((None, None, rnn_cell_num))
    reset_gate_val = tf.reshape(tf.concat(reset_gate_val, 1),\
                          (batch_size, -1, rnn_cell_num))
    update_gate_val = tf.reshape(tf.concat(update_gate_val, 1),\
                          (batch_size, -1, rnn_cell_num))
    reset_gate_val *= rnn_mask
    update_gate_val *= rnn_mask
    rnn_output *= rnn_mask


    return rnn_output, reset_gate_val, update_gate_val


def build_dynamic_rnn(
    rnn_type, rnn_cell_num, rnn_inputs, sequence_len, dropout_keep_prob):
    if rnn_type == 'gru':
        rnn_cell = local_cell.GRUCell(rnn_cell_num)
    elif rnn_type == 'lstm':
        rnn_cell = local_cell.BasicLSTMCell(rnn_cell_num)
    else:
        raise ValueError('RNN type should be LSTM or GRU')
    rnn_cell = local_cell.DropoutWrapper(rnn_cell, \
                input_keep_prob=dropout_keep_prob, \
                output_keep_prob=dropout_keep_prob)
    #LSTM: g1: forget gate; g2: input gate; g3: output gate
    #GRU:  g1: update gate; g2: reset gate; g3: zeros
    outputs, states, g1, g2, g3 = local_rnn.dynamic_rnn(\
                             rnn_cell, rnn_inputs, \
                             sequence_length=sequence_len, dtype=tf.float32)
    return outputs, g1, g2, g3


def reconstruction_loss(y, y_, mask):
    square_error = tf.multiply(tf.subtract(y_, y), tf.subtract(y_, y))
    square_error = tf.reduce_mean(square_error, reduction_indices=2)
    square_error *= mask

    sum_square_error = tf.reduce_sum(square_error, reduction_indices=1)
    sum_square_error /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(sum_square_error)


class NeuralNetwork(object):
    def __init__(self, tr_config, input_tensors):
        feature_dim = tr_config.feature_dim
        nn_hidden_num = tr_config.nn_hidden_num
        rnn_cell_num = tr_config.rnn_cell_num
        rnn_type = tr_config.rnn_type
        max_len = tr_config.max_len
        init_lr = tr_config.learning_rate
        tr_dropout = tr_config.dropout_keep_prob
        tr_zoneout = tr_config.zoneout_keep_prob
        max_epoch = tr_config.max_epoch
        tr_batch = tr_config.batch_size

        x = input_tensors["x"]
        y_ = input_tensors["y_"]
        batch_size = input_tensors["batch_size"]
        add_noise = input_tensors["add_noise"]
        dropout_keep_prob = input_tensors["dropout_keep_prob"]
        zoneout_keep_prob = input_tensors["zoneout_keep_prob"]

        rnd = tf.random_uniform(tf.shape(x))
        noise_th = tr_config.noise_magnitude * tf.ones_like(x, tf.float32)
        noise_mask = tf.ceil(tf.subtract(rnd, noise_th))

        model_input = tf.cond(add_noise, lambda: tf.multiply(x, noise_mask), lambda: x)
        mask = tf.sign(tf.reduce_max(tf.abs(y_), reduction_indices=2))
        sequence_len = tf.reduce_sum(mask, reduction_indices=1)
        with tf.variable_scope('autoencoder_model'):
            rnn_mask = tf.expand_dims(mask, 2)
            rnn_mask = tf.tile(rnn_mask, [1, 1, rnn_cell_num])
            with tf.variable_scope('encoder'):
                dnn_input = tf.reshape(model_input, [-1, feature_dim])
                dnn_output = build_dnn(dnn_input, feature_dim, nn_hidden_num)
                rnn_input = tf.reshape(dnn_output, [-1, max_len, nn_hidden_num])

                rnn_code, encoder_g1, encoder_g2, encoder_g3 = \
                 build_dynamic_rnn(rnn_type, rnn_cell_num, rnn_input, \
                  sequence_len, dropout_keep_prob)


            with tf.variable_scope('decoder'):
                rnn_output, decoder_g1, decoder_g2, decoder_g3 = \
                 build_dynamic_rnn(rnn_type, rnn_cell_num, rnn_code, \
                  sequence_len, dropout_keep_prob)

                dnn_input = tf.reshape(rnn_output, [-1, rnn_cell_num])
                dnn_output = build_dnn(dnn_input, rnn_cell_num, nn_hidden_num)

                W_output = tf.Variable(\
                 tf.truncated_normal([nn_hidden_num, feature_dim], stddev=0.1))
                b_output = tf.Variable(tf.constant(0.1, shape=[feature_dim]))
                decoder_output = tf.matmul(dnn_output, W_output) + b_output

            y = tf.reshape(decoder_output, [-1, max_len, feature_dim])


        self.re_loss = reconstruction_loss(y, y_, mask)

        reg_loss = tf.constant(0.)
        self.loss = self.re_loss
        lr_step = tf.Variable(0, trainable=False)
        decay_lr = tf.assign_add(lr_step, 1)
        self.lr = tf.train.exponential_decay(init_lr, lr_step, 1, 0.85)

        self.x = x
        self.y_ = y_
        self.batch_size = batch_size
        self.add_noise = add_noise
        self.dropout_keep_prob = dropout_keep_prob
        self.zoneout_keep_prob = zoneout_keep_prob

        self.tensor_list= {}
        self.tensor_list["re_loss"] = self.re_loss
        self.tensor_list['encoder_g1'] = encoder_g1
        self.tensor_list['encoder_g2'] = encoder_g2
        self.tensor_list['encoder_g3'] = encoder_g3
        self.tensor_list['decoder_g1'] = decoder_g1
        self.tensor_list['decoder_g2'] = decoder_g2
        self.tensor_list['decoder_g3'] = decoder_g3
        self.tensor_list['y'] = y

        self.saver = tf.train.Saver()


    def init_vars(self, sess):
        sess.run(tf.global_variables_initializer())


    def setup_train(self):
        self.train_all = tf.train.AdamOptimizer(\
         learning_rate=self.lr).minimize(self.loss)


    def train(self, sess, x, y_, dropout, zoneout, add_noise, batch_size):
        sess.run(self.train_all, feed_dict={self.x: x, self.y_: y_, \
                 self.batch_size: batch_size, self.add_noise: add_noise,\
                 self.dropout_keep_prob: dropout, \
                 self.zoneout_keep_prob: zoneout})


    def get_tensor_val(self, tensor_name, sess, x, y_, batch_size):
        tensor = self.tensor_list[tensor_name]
        return sess.run(tensor, feed_dict={self.x: x, self.y_: y_, \
                 self.batch_size: batch_size, self.add_noise: False,\
                 self.dropout_keep_prob: 1.0, \
                 self.zoneout_keep_prob: 1.0})

    def save_vars(self, sess, path):
        self.saver.save(sess, path)


    def restore_vars(self, sess, path):
        self.saver.restore(sess, path)






