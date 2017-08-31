#!/bin/bash

#script/00.train_lm.sh | tee log/00.train_lm.log
#script/01.format.sh | tee log/01.format.log
#script/02.extract.feat.sh | tee log/02.extract.feat.sh.log
#script/03.mono.train.sh | tee log/03.mono.train.log
#script/04a.01.mono.mkgraph.sh | tee log/04a.01.mono.mkgraph.log
#script/04a.02.mono.fst.sh | tee log/04a.02.mono.fst.log
#script/04b.mono.viterbi.sh | tee log/04b.mono.viterbi.log
#script/05.tree.build.sh | tee log/05.tree.build.log
script/06.tri.train.sh | tee log/06.tri.train.log
script/07a.01.tri.mkgraph.sh | tee log/07a.01.tri.mkgraph.log
script/07a.02.tri.fst.sh | tee log/07a.02.tri.fst.log
script/07b.tri.viterbi.sh | tee log/07b.tri.viterbi.log
