import subprocess
import sys
import numpy as np




def load_data_into_mem(scp_file):
    feat_proc = subprocess.Popen(['copy-feats scp:{} ark,t:- 2>/dev/null' \
                                  .format(scp_file)],stdout=subprocess.PIPE, \
                                  shell=True)
    data_list = []
    frame_list = []
    while True:
        line = feat_proc.stdout.readline().rstrip()
        if line == b'':
            feat_proc.terminate()
            break
        if b'[' in line:
            frame_list = []
            continue
        elif b']' in line:
           tmp = [float(x) for x in (line.split())[:-1]]
           frame_list.append(tmp)
           data_list.append(frame_list)
        else:
           tmp = [float(x) for x in (line.split())]
           frame_list.append(tmp)

    return data_list


def padding(data_list, max_len, feature_dim):
    X = np.zeros((len(data_list), max_len, feature_dim),
                 dtype = np.float32)
    for utt_idx in range(len(data_list)):
        for time_step_idx in range(len(data_list[utt_idx])):
            np.copyto(X[utt_idx][time_step_idx], \
                np.array(data_list[utt_idx][time_step_idx]))
    return X

def data_loader(scp_file, batch_size, \
                total_utt_num, max_len, feature_dim):
    feat_proc = subprocess.Popen(['copy-feats scp:{} ark,t:- 2>/dev/null' \
                                  .format(scp_file)],stdout=subprocess.PIPE, \
                                  shell=True)


    #initialization
    start = False
    utt_count = 0
    utt_num = 0
    sequence_idx = 0
    #arr_size is the size of the data this time, since the size of the last batch
    #do not have to be 'batch_size'
    remain_utt_num = total_utt_num - utt_num
    arr_size = min(batch_size, remain_utt_num)
    X = np.zeros((arr_size, max_len, feature_dim),
                 dtype = np.float32)
    sequence_len = np.zeros((arr_size))

    while True:
        line = feat_proc.stdout.readline().rstrip()

        if line == '' or utt_num >= total_utt_num:
            #end of the ark, close popoen process
            feat_proc.terminate()
            yield X
            break

        if b'[' in line :
            assert(start == False)
            start = True

            processed_uttID = (line.split())[0]
            continue

        if start == True and b']' not in line:
            #features
            for idx, s in enumerate(line.split()):
                X[utt_count][sequence_idx][idx] = float(s)
            sequence_idx += 1
            continue

        if b']' in line:
            #features
            for idx, s in enumerate(line[:-1].split()):
                X[utt_count][sequence_idx][idx] = float(s)

            #The end of a utterance, reset parameters
            start = False
            sequence_idx = 0
            utt_count += 1
            utt_num += 1

            if utt_count >= batch_size:
                if utt_num >= total_utt_num:
                    feat_proc.terminate()
                yield X
                utt_count = 0
                remain_utt_num = total_utt_num - utt_num
                arr_size = min(batch_size, remain_utt_num)
                X = np.zeros((arr_size, max_len, feature_dim),
                             dtype = np.float32)
                sequence_len = np.zeros((arr_size))


def get_specific_phn_bound(path):
    bounds = []
    with open(path,'r') as f:
        for line in f.readlines():
            a_bound = int(int(line.rstrip().split()[0]) / 160)
            if a_bound != 0:
                bounds.append(a_bound)

    return bounds


def bound_loader(bound_scp_file, batch_size):
    f = open(bound_scp_file, 'r')
    bounds_list = []
    counter = 0

    while True:
        line = f.readline()
        if line == '':
            yield bounds_list
            break

        if len(bounds_list) < batch_size:
            bound_file = line.rstrip().split()[1]
            bounds = get_specific_phn_bound(bound_file)
            bounds_list.append(bounds)

        if len(bounds_list) >= batch_size:
            yield bounds_list
            bounds_list = []
