'''
The config class
'''



class TrainConfig(object):
    def __init__(self, config_file):
        self.model_loc = 'models/my_model'
        self.max_len = -1
        self.feature_dim = -1

        with open(config_file, 'r') as f:
            for line in f.readlines():
                seg = line.rstrip().split()
                if seg[0] == 'Learning_rate':
                    self.learning_rate = float(seg[1])
                elif seg[0] == 'Dir':
                    self.dir = seg[1]
                elif seg[0] == 'Model':
                    self.model_name = seg[1]
                elif seg[0] == 'Hidden_neuron_num':
                    self.nn_hidden_num = int(seg[1])
                elif seg[0] == 'RNN_type':
                    self.rnn_type = seg[1]
                elif seg[0] == 'RNN_cell_num':
                    self.rnn_cell_num = int(seg[1])
                elif seg[0] == 'Batch_size':
                    self.batch_size = int(seg[1])
                elif seg[0] == 'Noise_magnitude':
                    self.noise_magnitude = float(seg[1])
                elif seg[0] == 'Dropout_keep_prob':
                    self.dropout_keep_prob = float(seg[1])
                elif seg[0] == 'Zoneout_keep_prob':
                    self.zoneout_keep_prob = float(seg[1])
                elif seg[0] == 'Max_epoch':
                    self.max_epoch = int(seg[1])


    def show_config(self):
        print('=============================================================')
        print('                    Training Config Info.                    ')
        print('=============================================================')
        print('Dir                           ', self.dir)
        print('Model                         ', self.model_name)
        print('Model_loc                     ', self.model_loc)
        print('Hidden_neuron_num             ', self.nn_hidden_num)
        print('RNN_cell_num                  ', self.rnn_cell_num)
        print('RNN_type                      ', self.rnn_type)
        print('Learning_rate                 ', self.learning_rate)
        print('Dropout_keep_prob             ', self.dropout_keep_prob)
        print('Zoneout_keep_prob             ', self.zoneout_keep_prob)
        print('Noise_magnitude               ', self.noise_magnitude)
        print('Batch_size                    ', self.batch_size)
        print('Max_epoch                     ', self.max_epoch)
        print ('')




class DecodeConfig(object):
    def __init__(self, config_file):
        self.model_loc = 'models/my_model'

        #1 for 10 ms, 2 for 20 ms
        self.tolerance_window = 2

        with open(config_file, 'r') as f:
            for line in f.readlines():
                seg = line.rstrip().split()
                if seg[0] == 'Batch_size':
                    self.batch_size = int(seg[1])
                if seg[0] == 'Dir':
                    self.dir = seg[1]
                elif seg[0] == 'Tolerance_window':
                    self.tolerance_window = int(seg[1])


    def show_config(self):
        print('=============================================================')
        print('                  Decode config info.                        ')
        print('=============================================================')
        print('Dir                           ', self.dir)
        print('Batch_size                    ', self.batch_size)
        print('Model_loc                     ', self.model_loc)
        print('Tolerance_window              ', self.tolerance_window)
