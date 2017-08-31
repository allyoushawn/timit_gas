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

./script/02.extract.feat.sh

for target in train test; do
    feat_scp=$feat_loc/${target}.39.cmvn.scp

    sort -k 1 -V ${feat_scp} > ${target}.scp

    rm -f ${target}.len ${target}.remove_list
    mkdir -p ../feature_scp
    mv ${target}.scp ../feature_scp

done

./gen_bounds.sh

rm *.wav.scp

#split dev set from train set
cat ../feature_scp/train.scp | sed -e '1~80d' >train.scp
cat ../feature_scp/train.scp | sed -n '1~80p' >dev.scp
cat ../bounds/phn/train.phn | sed -e '1~80d' >train.phn
cat ../bounds/phn/train.phn | sed -n '1~80p' >dev.phn

mv *.scp ../feature_scp/
mv *.phn ../bounds/phn/
