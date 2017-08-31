#!/bin/bash

if [ -f prepare_timit/setup.sh ]; then
    . prepare_timit/setup.sh
else
    echo "ERROR: prepare_timit/setup.sh is missing!"
    exit 1
fi

rnn_type="gru"
dir="${rnn_type}_grnn_32_dropout"

nn_hidden_num=64
rnn_cell_num=32
loss_fn="norm2"
model="${rnn_type}_autoencoder"
dropout_keep=0.3
zoneout_keep=1.0
lr=0.0008
noise=0.0
batch_size=60
max_epoch=60
gpu_device=0

#Decoding
decode_batch_size=25
tol_window=2

test_utt_num=$(cat feature_scp/test.scp | wc -l)
utt_num=$(cat feature_scp/train.scp | wc -l)

mkdir -p $dir
rm -rf ${dir}/*
cp -r rnn_autoencoder.py decode.py utils local_rnn feature_scp bounds $dir


cd $dir
mkdir models
rm -f info
echo "Dir                  $dir"  >>info
echo "Model                $model" >>info
echo "Structure            1 + 1 + 1 + 1" >>info
echo "Hidden_neuron_num    $nn_hidden_num" >>info
echo "RNN_type             $rnn_type" >>info
echo "RNN_cell_num         $rnn_cell_num" >>info
echo "Loss_function        $loss_fn" >>info
echo "Learning_rate        $lr" >>info
echo "Dropout_keep_prob    $dropout_keep" >>info
echo "Zoneout_keep_prob    $zoneout_keep" >>info
echo "Noise_magnitude      $noise" >>info
echo "Tolerance_window     $tol_window" >>info
echo "Batch_size           $batch_size" >>info
echo "Max_epoch            $max_epoch" >>info
echo "Train_utt_num        $utt_num" >>info
echo ""

CUDA_VISIBLE_DEVICES=${gpu_device} ./rnn_autoencoder.py  2>&1 | tee train.log


rm -f decode_info
echo "Dir                  $dir"  >>decode_info
echo "Batch_size           $decode_batch_size" >>decode_info
echo "Test_utt_num         $test_utt_num" >>decode_info
echo "Tolerance_window     $tol_window"  >>decode_info
echo ""

rm -f testing.result
CUDA_VISIBLE_DEVICES=$gpu_device ./decode.py | tee testing.result

