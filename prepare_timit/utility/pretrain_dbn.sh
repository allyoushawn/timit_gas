#!/bin/bash
# Copyright 2013-2015 Brno University of Technology (author: Karel Vesely)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# To be run from ../../
#
# Restricted Boltzman Machine (RBM) pre-training by Contrastive Divergence 
# algorithm (CD-1). A stack of RBMs forms a Deep Belief Neetwork (DBN).
#
# This script by default pre-trains on plain features (ie. saved fMLLR features), 
# building a 'feature_transform' containing +/-5 frame splice and global CMVN.
#
# There is also a support for adding speaker-based CMVN, deltas, i-vectors,
# or passing custom 'feature_transform' or its prototype.
# 

# Begin configuration.

# topology, initialization,
nn_depth=6             # number of hidden layers,
hid_dim=2048           # number of neurons per layer,
param_stddev_first=0.1 # init parameters in 1st RBM
param_stddev=0.1 # init parameters in other RBMs
input_vis_type=gauss # type of visible nodes on DBN input

# number of iterations,
rbm_iter=1            # number of pre-training epochs (Gaussian-Bernoulli RBM has 2x more)

# pre-training opts,
rbm_lrate=0.4         # RBM learning rate
rbm_lrate_low=0.01    # lower RBM learning rate (for Gaussian units)
rbm_l2penalty=0.0002  # L2 penalty (increases RBM-mixing rate)
rbm_extra_opts=

# data processing,
copy_feats=true     # resave the features to tmpdir,
copy_feats_tmproot=/tmp/kaldi.XXXX # sets tmproot for 'copy-feats',

# feature processing,
context=5            # (default) splice features both-ways along time axis,
cmvn_opts=          # (optional) adds 'apply-cmvn' to input feature pipeline, see opts,
delta_opts=         # (optional) adds 'add-deltas' to input feature pipeline, see opts,
ivector=            # (optional) adds 'append-vector-to-feats', it's rx-filename,

feature_transform_proto= # (optional) use this prototype for 'feature_transform',
feature_transform=  # (optional) directly use this 'feature_transform',

# misc.
verbose=1 # enable per-cache reports
skip_cuda_check=false

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f setup.sh ] && . ./setup.sh;
. utility/parse_options.sh || exit 1;

set -euo pipefail

dir=$1


echo "# INFO #"
echo "  Pre-training Deep Belief Network as a stack of RBMs"
echo "  dir: $dir"
echo "  feature_transform: $feature_transform"
echo "  dbn_depth: $nn_depth"
echo "  rbm_iter: $rbm_iter"

[ -e $dir/${nn_depth}.dbn ] && echo "$0 Skipping, already have $dir/${nn_depth}.dbn" && exit 0

# check if CUDA compiled in and GPU is available,
if ! $skip_cuda_check; then cuda-gpu-available || exit 1; fi

mkdir -p $dir/log

###### PREPARE FEATURE PIPELINE ######
# read the features
feats_tr=$train_feat_setup #dirty code
splice_feat_train="ark:$train_feat_mlp splice-feats --print-args=false --left-context=$context --right-context=$context scp:- ark:- |"
# get feature dim,
feat_dim=$(feat-to-dim "ark:$feats_tr" - 2>${dir}/log/feat.to.dim.log)
echo "#  feature dim: $feat_dim (input of 'feature_transform')"
# Now we start building 'feature_transform' which goes right in front of a NN. 
# The forwarding is computed on a GPU before the frame shuffling is applied.
#
# Same GPU is used both for 'feature_transform' and the NN training.
# So it has to be done by a single process (we are using exclusive mode).
# This also reduces the CPU-GPU uploads/downloads to minimum.


###### GET THE DIMENSIONS ######
num_fea=$(feat-to-dim --print-args=false "$splice_feat_train nnet-forward --use-gpu=no $feature_transform ark:- ark:- |" - 2>/dev/null)  #dirty code
echo "num_fea is "$num_fea
num_hid=$hid_dim


###### PERFORM THE PRE-TRAINING ######
for depth in $(seq 1 $nn_depth); do
  echo
  echo "# PRE-TRAINING RBM LAYER $depth"
  RBM=$dir/$depth.rbm
  [ -f $RBM ] && echo "RBM '$RBM' already trained, skipping." && continue

  # The first RBM needs special treatment, because of Gussian input nodes,
  if [ "$depth" == "1" ]; then
    # This is usually Gaussian-Bernoulli RBM (not if CNN layers are part of input transform)
    # initialize,
    echo "# initializing '$RBM.init'"
    echo "<Rbm> <InputDim> $num_fea <OutputDim> $num_hid <VisibleType> $input_vis_type <HiddenType> bern <ParamStddev> $param_stddev_first" > $RBM.proto
    nnet-initialize $RBM.proto $RBM.init 2>$dir/log/nnet-initialize.$depth.log || exit 1
    # pre-train,
    num_iter=$rbm_iter; [ $input_vis_type == "gauss" ] && num_iter=$((2*rbm_iter)) # 2x more epochs for Gaussian input
    [ $input_vis_type == "bern" ] && rbm_lrate_low=$rbm_lrate # original lrate for Bernoulli input
    echo "# pretraining '$RBM' (input $input_vis_type, lrate $rbm_lrate_low, iters $num_iter)"
    rbm-train-cd1-frmshuff --learn-rate=$rbm_lrate_low --l2-penalty=$rbm_l2penalty \
      --num-iters=$num_iter --verbose=$verbose \
      --feature-transform=$feature_transform \
      $rbm_extra_opts \
      $RBM.init "$splice_feat_train" $RBM 2>$dir/log/rbm.$depth.log || exit 1
  else
    # This is Bernoulli-Bernoulli RBM,
    # cmvn stats for init,
    echo "# computing cmvn stats '$dir/$depth.cmvn' for RBM initialization"
    if [ ! -f $dir/$depth.cmvn ]; then 
      nnet-forward --print-args=false --use-gpu=yes \
        "nnet-concat $feature_transform $dir/$((depth-1)).dbn - |" \
        "$(echo $splice_feat_train | sed 's|train.scp|train.scp.10k|')" ark:- | \
      compute-cmvn-stats --print-args=false ark:- - | \
      cmvn-to-nnet --print-args=false - $dir/$depth.cmvn || exit 1
    else
      echo "# compute-cmvn-stats already done, skipping."
    fi
    # initialize,
    echo "initializing '$RBM.init'"
    echo "<Rbm> <InputDim> $num_hid <OutputDim> $num_hid <VisibleType> bern <HiddenType> bern <ParamStddev> $param_stddev <VisibleBiasCmvnFilename> $dir/$depth.cmvn" > $RBM.proto
    nnet-initialize $RBM.proto $RBM.init 2>$dir/log/nnet-initialize.$depth.log || exit 1
    # pre-train,
    echo "pretraining '$RBM' (lrate $rbm_lrate, iters $rbm_iter)"
    rbm-train-cd1-frmshuff --learn-rate=$rbm_lrate --l2-penalty=$rbm_l2penalty \
      --num-iters=$rbm_iter --verbose=$verbose \
      --feature-transform="nnet-concat $feature_transform $dir/$((depth-1)).dbn - |" \
      $rbm_extra_opts \
      $RBM.init "$splice_feat_train" $RBM 2>$dir/log/rbm.$depth.log || exit 1
  fi

  # Create DBN stack,
  if [ "$depth" == "1" ]; then
    echo "# converting RBM to $dir/$depth.dbn"
    rbm-convert-to-nnet $RBM $dir/$depth.dbn
  else 
    echo "# appending RBM to $dir/$depth.dbn"
    nnet-concat $dir/$((depth-1)).dbn "rbm-convert-to-nnet $RBM - |"  $dir/$depth.dbn
  fi

done

echo
echo "# REPORT"
echo "# RBM pre-training progress (line per-layer)"
grep progress $dir/log/rbm.*.log
echo 

echo "Pre-training finished."

sleep 3
exit 0
