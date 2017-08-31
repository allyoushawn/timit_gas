#!/bin/bash

. setup.sh
bash script/01.format.sh 2>&1
bash script/02.extract.feat.sh  2>&1 | tee log/mk_mfcc.log || exit 1
bash script/03.mono.train.sh 2>&1 | tee log/mono_train.log || exit 1
bash script/04a.01.mono.mkgraph.sh 2>&1 | tee log/mono_mkgraph.log || exit 1
bash script/05.tree.build.sh 2>&1 | tee log/tree_build.log || exit 1
bash script/06.tri.train.sh 2>&1 | tee log/tri_train.log || exit 1
bash script/07a.01.tri.mkgraph.sh 2>&1 | tee log/tri_mkgraph.log || exit 1
bash script/07a.02.tri.fst.sh 2>&1 | tee log/tri_decode.log || exit 1
#bash script/08.mlp.train.sh | tee log/mlp_train.log || exit 1
#bash script/09.mlp.decode.sh | tee log/mlp_decode.log || exit 1

./align_dev.sh

ali_dir=ali
model_dir=exp/tri

for target in train dev; do
    ali-to-pdf ${model_dir}/final.mdl ark:${ali_dir}/${target}.ali ark,t:${target}.label
done
