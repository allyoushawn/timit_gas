#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run.sh <type>"
    exit 1
fi

command_type=$1

if [ $command_type != "train" -a $command_type != "decode" ]; then
    echo "Command type should be \"decode\" "
    exit 1
fi


dir="/media/hdd/csie/exp_results/gru_ae_32_dropout_2nd_decode_every_epoch"
batch_size=50
tol_window=2
gpu_device=0
test_utt_num=$(cat feature_scp/test.scp | wc -l)
max_epoch=40
if [ $command_type = "decode" ]; then
    if [ ! -d $dir ]; then 
        echo "No such directory: $dir"
	exit 1
    fi
    cp -r decode.py utils local_rnn feature_scp bounds  $dir
    cd $dir

    for decode_model_num in $(seq $max_epoch); do

	rm -f decode_info
	echo "============================================================="
	echo "                      Config Info.                           "
	echo "============================================================="
	echo "Dir                  $dir"
	echo "Model_number         $decode_model_num"  | tee -a decode_info
	echo "Batch_size           $batch_size" | tee -a decode_info
	echo "Test_utt_num         $test_utt_num" | tee -a decode_info
	echo "Tolerance_window     $tol_window" | tee -a decode_info
	echo ""
	echo "============================================================="
	echo "                    Decoding Process                         "
	echo "============================================================="

	rm -f model_${decode_model_num}.result
	CUDA_VISIBLE_DEVICES=$gpu_device ./decode.py | tee model_${decode_model_num}.result

    done
fi

