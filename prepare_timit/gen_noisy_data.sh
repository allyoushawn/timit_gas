#!/bin/bash

if [ -f setup.sh ]; then
    . setup.sh
else
    echo "ERROR: setup.sh is missing!"
    exit 1
fi

echo "Prepare correct wav.scp"
for target in train test; do
    rm -f ${target}.wav.scp
    while read line; do
        echo $line | sed "s#${old_kaldi_root}#${kaldi_root}#g
                           s#${old_timit_root}#${timit_root}#g" >> ${target}.wav.scp
    done < material/${target}.wav.scp
done

echo "Prepare noisy data"
for target in train test; do
    rm -f noisy_${target}.wav.scp
    while read -ra line_seg; do
        wav_file=${line_seg[4]}
        utt_id=${line_seg[0]}
        wav_dir=$( echo $(dirname $wav_file) | sed "s#${timit_root}#${noisy_timit_root}#g")
        noisy_wav_file=$( echo ${wav_file} | sed "s#${timit_root}#${noisy_timit_root}#g")

        if [ ! -d ${wav_dir} ]; then
            mkdir -p ${wav_dir}
        fi
        sox ${wav_file} -p synth whitenoise vol 0.002 | sox -m ${wav_file} - ${noisy_wav_file}
        echo "${utt_id} ${noisy_wav_file}" >>noisy_${target}.wav.scp


    done <${target}.wav.scp
    mv noisy_${target}.wav.scp ${target}.wav.scp
done

./script/02.extract.feat.sh

for target in train test; do
    feat_scp=$feat_loc/${target}.39.cmvn.scp
    #feat_scp=$feat_loc/${target}.39.scp
    #feat_scp=$feat_loc/${target}.13.scp
    feat-to-len scp:$feat_scp ark,t:${target}.len
    if [ $target == "train" ]; then
        remove_num=40
    else
        remove_num=25
    fi
    cat ${target}.len | sort -k 2  -V | tail -n $remove_num >${target}.remove_list

    remove_list=()
    counter=0
    while IFS=' ' read -ra line; do
        remove_list[$counter]=${line[0]}
        counter=$(( $counter + 1))
    done < ${target}.remove_list

    rm -f ${target}.scp
    while read -ra line_seg; do
        flag="no"
        for element in ${remove_list[@]}; do
            if [ $element == ${line_seg[0]} ];then
                flag="yes"
                break
            fi
        done
        if [ $flag == "no" ]; then
            echo "${line_seg[0]} ${line_seg[1]}" >>${target}.scp
        fi

    done <$feat_scp

    sort -k 1 -V ${target}.scp > sorted_${target}.scp

    rm -f ${target}.len ${target}.remove_list
    mv sorted_${target}.scp ${target}.scp
    mkdir -p ../feature_scp
    mv ${target}.scp ../feature_scp

done

rm *.wav.scp

#split dev set from train set
cat ../feature_scp/train.scp | sed -e '1~80d' >train.scp
cat ../feature_scp/train.scp | sed -n '1~80p' >dev.scp
cat ../bounds/phn/train.phn | sed -e '1~80d' >train.phn
cat ../bounds/phn/train.phn | sed -n '1~80p' >dev.phn

mv *.scp ../feature_scp/
