#!/bin/bash

if [ -f setup.sh ]; then
	. setup.sh;
else
	echo "ERROR: setup.sh is missing!";
	exit 1;
fi

if [ ! -s exp/tri/dev.ali ]; then
  bash align_dev.sh
fi

echo "====================================================================================="
echo "                                    MLP  Training                                    "
echo "====================================================================================="


###################### Start Config ######################
nnet_type=dnn

dir=exp/mlp
hmmdir=exp/tri
alidir=exp/tri
mkdir -p $dir
mkdir -p $dir/log
mkdir -p $dir/nnet


#NN parameter part
depth=2
rbm_depth=0
dbn=""
rbm_hidd_num=1024
num_hidden=1024
minibatch_size=256
randomizer_size=32768
randomizer_seed=777


#LSTM parameter
lstm_depth=2
lstm_cell_num=1024
lstm_recurrent_num=512


#training parameters
max_iters=50
min_iters=15
keep_lt_iters=10
learn_rate=0.008
start_halving_impr=0.01
end_halving_impr=0.001
halving_factor=0.5

#feature part
feat_train=$train_feat_mlp
feat_dev=$dev_feat_mlp
feat_test=$test_feat_mlp
context=5
splice_feat_train="ark:$feat_train splice-feats --print-args=false --left-context=$context --right-context=$context scp:- ark:- |"
splice_feat_dev="ark:$feat_dev splice-feats --print-args=false --left-context=$context --right-context=$context scp:- ark:- |"
labels_tr="ark:ali-to-pdf $alidir/final.mdl ark:$alidir/train.ali ark:- | ali-to-post ark:- ark:- |"
labels_dev="ark:ali-to-pdf $alidir/final.mdl ark:$alidir/dev.ali ark:- | ali-to-post ark:- ark:- |"
feature_transform=$dir/norm.nnet

copy-transition-model --binary=false $hmmdir/final.mdl $dir/final.mdl 2> $dir/log/copy-transition-model.log || exit 1
cp $hmmdir/tree $dir/tree

# get pdf-counts, used later for decoding/aligning,
labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl ark:$alidir/train.ali ark:- |"
#analyze-counts --verbose=1 --counts-dim=$state_number --binary=false "$labels_tr_pdf" $dir/ali_train_pdf.counts 2>$dir/log/analyze_counts_pdf.log || exit 1
analyze-counts --verbose=1 --binary=false "$labels_tr_pdf" $dir/ali_train_pdf.counts 2>$dir/log/analyze_counts_pdf.log || exit 1

feat-to-dim --print-args=false "$splice_feat_train" $dir/dim 2> $dir/log/feat.to.dim.log || exit 1
num_feat=`cat $dir/dim`
num_target=`hmm-info --print-args=false $dir/final.mdl | grep pdfs | awk '{ print $NF }'`

###################### End of Config ######################


log=$dir/log/norm.nnet.log
echo ""
echo "Computing normalization for input layer"
echo "    output -> $dir/norm.nnet"
echo "    log -> $log"

if [ -s  $dir/norm.nnet ]; then
  echo "$dir/norm.nnet exist! skipping..."
else
  ( compute-cmvn-stats --binary=false "$splice_feat_train" - | cmvn-to-nnet - $feature_transform ) 2> $log || exit 1
fi
  


###################### DBN Part ######################
if [ $rbm_depth -ge 1 ]; then
  rbm_dir=exp/rbm
  mkdir -p $rbm_dir
  if [ ! -s ${rbm_dir}/${rbm_depth}.dbn ]; then
    utility/pretrain_dbn.sh --nn_depth $rbm_depth --hid-dim $rbm_hidd_num --rbm-iter 10   --feature-transform $feature_transform $rbm_dir
  else
    echo "${rbm_depth}.dbn exists, skips..."
  fi
  dbn=${rbm_dir}/${rbm_depth}.dbn
fi


###################### NN Part ######################
echo ""
echo "Network topology :"
echo "    NN depth = [ $depth ]"
[ ! -z $dbn ] && echo "    DBN depth = [ $rbm_depth ]"
echo "    feature dimension = [ $num_feat ]"
echo "    number of nodes in one hidden layer = [ $num_hidden ]"
echo "    number of targets = [ $num_target ]"
echo ""
echo "Iteration 00 :"
echo "    generating network prototype"
echo "        output -> $dir/nnet.proto"
nnet_proto=$dir/nnet.proto

case "$nnet_type" in
  dnn)
    ## if there is no $dbn here will have bug. ##
    if [ ! -z $dbn ];then
      get_dim_from="nnet-concat $feature_transform $dbn - |" 
      num_feat=$(feat-to-dim "$splice_feat_train nnet-forward \"$get_dim_from\" ark:- ark:- |" -)
    fi

    utility/make_nnet_proto.py $proto_opts \
        ${bn_dim:+ --bottleneck-dim=$bn_dim} \
        $num_feat $num_target $depth $num_hidden >$nnet_proto || exit 1 
    train_tool='nnet-train-frmshuff'
    train_opts="--minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomizer-seed=$randomizer_seed"
    ;;
  lstm)
    if [ ! -z $dbn ];then
      get_dim_from="nnet-concat $feature_transform $dbn - |"
      num_feat=$(feat-to-dim "$splice_feat_train nnet-forward \"$get_dim_from\" ark:- ark:- |" -)
    fi

    utility/make_lstm_proto.py $proto_opts \
       $num_feat $num_target --num-cells=$lstm_cell_num --num-layers=$lstm_depth --num-recurrent=$lstm_recurrent_num >$nnet_proto || exit 1
    train_tool='nnet-train-lstm-streams'
    train_opts="--minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomizer-seed=$randomizer_seed"
    ;;
esac
echo ""
echo "Summary: We use $nnet_type neural network, and $train_tool as our training tool."

nnet_init=$dir/nnet.init;
log=$dir/log/nnet.init.log
echo ""
echo "    initializing the NN '$nnet_proto' -> '$nnet_init'"
echo "    log -> $log"
nnet-initialize --seed=$randomizer_seed $nnet_proto $nnet_init 2>$log || exit 1

if [ $rbm_depth -ge 1 ]; then
  nnet_init_old=$nnet_init; nnet_init=$dir/nnet_$(basename $dbn)_dnn.init
  nnet-concat $dbn $nnet_init_old $nnet_init || exit 1
fi

nnet_best=$nnet_init

log=$dir/log/iter00.cv.log

echo ""
echo "    Run the first cross validation."
echo "    log -> $log"

$train_tool --cross-validate=true --randomize=false --use-gpu=yes  $train_opts\
  --feature-transform=$feature_transform "$splice_feat_dev" "$labels_dev" $nnet_best \
  2>$log || exit 1

cv_loss=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
loss_type=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')
echo "    CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $cv_loss) $loss_type"

iter=1
halving=0
timer=$SECONDS
while [ $iter -le $max_iters ]; do
  x=`printf "%02g" $iter` 
  echo ""
  echo "Epoch $x :"
  timer=$SECONDS
  log=$dir/log/iter${x}.tr.log
  nnet_next=$dir/nnet/nnet_iter$x
  echo ""
  echo "    [Training] "
  echo "      log -> $log"

  $train_tool --cross-validate=false --randomize=true --use-gpu=yes --learn-rate=$learn_rate $train_opts \
    --feature-transform=$feature_transform \
    "$splice_feat_train" "$labels_tr" $nnet_best $nnet_next \
    2>$log || exit 1
  tr_loss=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo  "      TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)) "



  log=$dir/log/iter${x}.cv.log
  echo ""
  echo "    [Cross Validation]"
  echo "      log -> $log"
  $train_tool --cross-validate=true --randomize=false --use-gpu=yes  $train_opts \
    --feature-transform=$feature_transform "$splice_feat_dev" "$labels_dev" $nnet_next \
    2>$log || exit 1
  loss_new=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo -n "      CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "


  if [ $iter -eq 1 ]; then loss=$tr_loss; fi

  #accept the training result or not?
  loss_prev=$cv_loss
  if [ 1 == $(bc <<< "$loss_new < $cv_loss") -o $iter -le $keep_lt_iters -o $iter -le $min_iters ]; then
    # accepting: the loss was better, or we had fixed learn-rate, or we had fixed epoch-number,
    cv_loss=$loss_new
    nnet_best=$nnet_next
    echo "    nnet accepted ($(basename $nnet_best))"

  else
    mv $nnet_next ${nnet_next}_rejected
    echo "    nnet rejected ($(basename $nnet_next))"
  fi
  
  timer=$[$SECONDS-$timer];
  echo ""
  echo "    excution  time for epoch $x = `utility/timer.pl $timer`"

  if [ $iter -le $keep_lt_iters ]; then
    iter=$[$iter+1];
    continue
  fi

  # stopping criterion,
  rel_impr=$(bc <<< "scale=10; ($loss_prev-$cv_loss)/$loss_prev")
  if [ 1 == $halving -a 1 == $(bc <<< "$rel_impr < $end_halving_impr") ]; then
    if [ $iter -le $min_iters ]; then
      echo "    we were supposed to finish, but we continue as min_iters : $min_impr"
      iter=$[$iter+1];
      continue
    fi
    echo ""
    echo "    finished, too small rel. improvement $rel_impr"
    break
  fi

  # start learning-rate fade-out when improvement is low,
  if [ 1 == $(bc <<< "$rel_impr < $start_halving_impr") ]; then
    halving=1
    echo ""
    echo "    Start to halving learning rate..."
    echo $halving >$dir/.halving
  fi

  # reduce the learning-rate,
  if [ 1 == $halving ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    echo $learn_rate >$dir/.learn_rate
  fi

  iter=$[$iter+1];
done


#Training Done, select the best NN.
if [ $nnet_best != $nnet_init ];then
  nnet_final=${nnet_best}_final_
  ( cd $dir/nnet; ln -sf $(basename $nnet_best) $(basename $nnet_final); )
  ( cd $dir; ln -sf nnet/$(basename $nnet_final) final.nnet; )
  echo ""
  echo "Succeeded training the Neural Network : $dir/final.nnet"
else
  echo "Error training neural network..."
  exit 1
fi

echo ""
echo "Execution time for whole script = `utility/timer.pl $SECONDS`"
echo ""
