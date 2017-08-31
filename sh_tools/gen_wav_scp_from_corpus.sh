#!/bin/bash

timit=/media/hdd/csie/corpus/timit
kaldi_tool_header="/home/allyoushawn/Kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav"

for target in train test; do
    rm -f ${target}.wav.scp
    dir=${timit}/${target}
    for file in $(find $dir | grep .wav); do

        while IFS='/' read -ra seg; do

	    while IFS='.' read -ra seg2; do
	        echo "${seg[8]}_${seg2[0]} $kaldi_tool_header $file |" >>${target}.wav.scp
	    done<<<${seg[9]}
	done <<< $file
        
    done
done
