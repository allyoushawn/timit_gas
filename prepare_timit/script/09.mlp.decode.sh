#!/bin/bash
# Based on Kaldi-timit-s5 code:steps/nnet/decode.sh
# Usage Ex: decode.sh --acwt 0.2 exp/tri3/graph data/test decode_dnn/test

if [ -f setup.sh ]; then
	. setup.sh
else
	echo "ERROR: setup.sh is missing!"
	exit 1
fi

# Begin configuration section. 
nnet=exp/mlp/final.nnet
model=exp/tri/final.mdl
HCLG=graph/tri/HCLG.fst
class_frame_counts=exp/mlp/ali_train_pdf.counts
dir=fst/mlp
feature_transform=exp/mlp/norm.nnet #The feature_transform here we use only simple cmvn.

acwt=0.10 # note: only really affects pruning (scoring is on lattices).
beam=13.0
lattice_beam=8.0
context=5

min_active=200
max_active=7000 # limit of active tokens
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes

nnet_forward_opts="--no-softmax=true --prior-scale=1.0"
use_gpu="yes" # yes|no|optionaly

thread_string=""
num_threads=$cpu_num
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

# End configuration section.

mkdir -p $dir/log

# Create the feature stream,
feat_dev=$dev_feat_mlp
feat_test=$test_feat_mlp
splice_feat_dev="ark:$feat_dev splice-feats --print-args=false --left-context=$context --right-context=$context scp:- ark:- |"
splice_feat_test="ark:$feat_test splice-feats --print-args=false --left-context=$context --right-context=$context scp:- ark:- |"

timer=$SECONDS
max_acc=0.0
opt_acwt=none
echo ""
echo "=================================Finding the optimal acoustic weight================================"
for x in `seq 1 10`; do
	acwt=`perl -e "printf (\"%4.2f\", $x * 0.02);"`
	log=$dir/log/dev.${acwt}.log
	echo "====================================================================================================="
	echo "          Generating char level transcription on acoustic weight = $acwt -> $dir/dev.${acwt}.tra"
	echo "          Then evaluate the accuracy."
	echo "          log -> $log"
	echo "====================================================================================================="
	
	nnet-forward $nnet_forward_opts --feature-transform=$feature_transform --class-frame-counts=$class_frame_counts \
		 --use-gpu=$use_gpu "$nnet" "$splice_feat_dev" ark:-  | \
	latgen-faster-mapped$thread_string --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam \
		--lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=decode/words.txt \
		$model $HCLG ark:- ark:$dir/dev.${acwt}.lat ark,t:$dir/dev.${acwt}.tra ark,t:$dir/dev.${acwt}.ali  2>$log || exit 1;

	cat $dir/dev.${acwt}.tra \
	  | utility/int2sym.pl --ignore-first-field decode/words.txt \
	  | python utility/word2char.py \
	  >$dir/dev.${acwt}.rec
	cat $dir/dev.${acwt}.rec \
	  | python utility/compute-acc.py decode/dev.text \
	  >$dir/dev.${acwt}.acc

	acc=`grep "overall accuracy" $dir/dev.${acwt}.acc | awk '{ print $4 }'`

	if [ "1" == `awk "BEGIN{print($acc>$max_acc)}"` ]; then
        opt_acwt=$acwt
        max_acc=$acc
  	fi
  
	echo "    acoustic weight = [ $acwt ]"
	echo "    result -> $dir/dev.$acwt.rec"
  	echo "    accuracy -> [ $acc ]%"
        timer=$[$SECONDS-$timer];
        echo "    excution  time for the evaluation on $x = `utility/timer.pl $timer`"

done
timer=$[$SECONDS-$timer]
echo "Execution time for finding the optimal weight = `utility/timer.pl $timer`"
echo "Optimal acoustic weight = $opt_acwt"

echo ""
echo "Testing set evaluation:"
log=$dir/log/test.log
echo "    log->$log"

	nnet-forward $nnet_forward_opts --feature-transform=$feature_transform --class-frame-counts=$class_frame_counts \
                 --use-gpu=$use_gpu "$nnet" "$splice_feat_test" ark:-  | \
        latgen-faster-mapped$thread_string --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam \
                --lattice-beam=$lattice_beam --acoustic-scale=$opt_acwt --allow-partial=true --word-symbol-table=decode/words.txt \
                $model $HCLG ark:- ark:$dir/test.lat ark,t:$dir/test.tra ark,t:$dir/test.ali  2>$log || exit 1;

        cat $dir/test.tra \
          | utility/int2sym.pl --ignore-first-field decode/words.txt \
          | python utility/word2char.py \
          >$dir/test.rec
        cat $dir/test.rec \
          | python utility/compute-acc.py decode/test.text \
          >$dir/test.acc

        acc=`grep "overall accuracy" $dir/test.acc | awk '{ print $4 }'`
        echo ""
        echo "    result -> $dir/test.rec"
        echo "    accuracy -> [ $acc ]%"

timer=$[$SECONDS-$timer]
echo ""
echo "Execution time for test set decoding = `utility/timer.pl $timer`"

echo ""
echo "Execution time for whole script = `utility/timer.pl $SECONDS`"
echo ""

exit 0;
