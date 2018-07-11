#!/bin/bash

#specify the followings
kaldi_root=/opt/kaldi
timit_root=/share/corpus/timit
export feat_loc=/home/allyoushawn/handover/timit_gas//timit_feat

#leave the followings untouched
old_kaldi_root=/home/allyoushawn/Kaldi
old_timit_root=/media/hdd/csie/corpus/timit

kaldi_src=$kaldi_root/src
openfst_root=$kaldi_root/tools/openfst
PATH=$openfst_root/bin:$PATH
PATH=$kaldi_src/bin:$PATH
PATH=$kaldi_src/fstbin/:$PATH
PATH=$kaldi_src/gmmbin/:$PATH
PATH=$kaldi_src/featbin/:$PATH
PATH=$kaldi_src/sgmmbin/:$PATH
PATH=$kaldi_src/sgmm2bin/:$PATH
PATH=$kaldi_src/fgmmbin/:$PATH
PATH=$kaldi_src/latbin/:$PATH
PATH=$kaldi_src/nnetbin/:$PATH
export PATH=$PATH


