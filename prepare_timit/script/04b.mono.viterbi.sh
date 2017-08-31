#!/bin/bash

if [ -f setup.sh ]; then
  . setup.sh;
else
  echo "ERROR: setup.sh is missing!";
  exit 1;
fi

srcdir=exp/mono
dir=viterbi/mono

lm=decode/lm.arpa.txt
#lm=test.lm
lex=decode/lexicon.txt
dev_feat=$dev_feat_setup
test_feat=$test_feat_setup

dev_acwt=0.10
dev_beam=13.0
test_beam=15.0

mkdir -p $dir
mkdir -p $dir/log
mkdir -p $dir/lat

echo "Converting acoustic models to HTK format"
if [ ! -f $dir/final.mmf ] || [ ! -f $dir/tiedlist ]; then
  log=$dir/log/am.to.htk.log
  echo "    output -> $dir/final.mmf $dir/tiedlist"
  echo "    log -> $log"
  vulcan-am-kaldi-to-htk --trans-mdl=$srcdir/final.mdl --tree=$srcdir/tree \
    --phonelist=train/phones.txt --htk-mmf=$dir/final.mmf --htk-tiedlist=$dir/tiedlist \
    2> $log
else
  echo "    $dir/final.mmf $dir/tiedlist exist , skipping ..."
fi

timer=$SECONDS
log=$dir/log/latgen.dev.log
echo "Generating lattices for dev set with acoustic weight = [ $dev_acwt ]"
echo "    output dir -> $dir/lat/"
echo "    log -> $log"
Hybrid.HDecode.mod --trace=1 --beam=$dev_beam \
  --am-weight=$dev_acwt --lm-weight=1.0 \
  --lat-dir=$dir/lat --mlf=$dir/dev.mlf \
  --htk-mmf=$dir/final.mmf --htk-tiedlist=$dir/tiedlist \
  --arpa-lm=$lm --lex=$lex --phonelist=train/phones.txt \
  --gmm-weight=1.0 --gmm-mdl=$srcdir/final.mdl --gmm-tree=$srcdir/tree \
  --feature="ark,s,cs:$dev_feat" \
  2> $log
timer=$[$SECONDS-$timer]
echo "    execution time for generating lattices for dev set = `utility/timer.pl $timer`"

timer=$SECONDS
max_acc=0.0
opt_acwt=none
echo "Recoring lattice to find optimal acoustic weight"
for x in `seq 1 10`; do
  scale=`perl -e "printf (\"%4.2f\", $x * 0.2);"`
  acwt=`perl -e "printf (\"%4.2f\", $dev_acwt * $scale);"`
  Hybrid.HLRescore --am-weight=$scale --mlf=$dir/dev.$acwt.result \
    --list=decode/dev.list --lat-dir=$dir/lat --lex=$lex \
    2> $dir/log/rescore.dev.$acwt.log
  cat $dir/dev.$acwt.result \
    | utility/result.htk2kaldi.pl \
    | python utility/word2char.py \
    > $dir/dev.$acwt.rec
  cat $dir/dev.$acwt.rec \
    | python utility/compute-acc.py decode/dev.text \
    > $dir/dev.$acwt.acc
  acc=`grep "overall accuracy" $dir/dev.$acwt.acc | awk '{ print $4 }'`
  if [ "1" == `awk "BEGIN{print($acc>$max_acc)}"` ]; then
      opt_acwt=`perl -e "printf (\"%4.2f\", $dev_acwt * $scale);"`
      max_acc=$acc
  fi
  echo "    acoustic weight = [ $acwt ]"
  echo "        result -> $dir/dev.$acwt.rec"
  echo "        accuracy -> [ $acc ] %"
done
timer=$[$SECONDS-$timer]
echo "    optimal acoustic weight = [ $opt_acwt ]"
echo "    corresponding accuracy = [ $max_acc ] %"
echo "    execution time for rescoring lattices = `utility/timer.pl $timer`"

timer=$SECONDS
log=$dir/log/latgen.test.log
echo "Generating results for test set with acoustic weight = [ $opt_acwt ]"
echo "    output -> $dir/test.mlf"
echo "    log -> $log"
Hybrid.HDecode.mod --trace=1 --beam=$test_beam \
  --am-weight=$opt_acwt --lm-weight=1.0 \
  --mlf=$dir/test.mlf \
  --htk-mmf=$dir/final.mmf --htk-tiedlist=$dir/tiedlist \
  --arpa-lm=$lm --lex=$lex --phonelist=train/phones.txt \
  --gmm-weight=1.0 --gmm-mdl=$srcdir/final.mdl --gmm-tree=$srcdir/tree \
  --feature="ark,s,cs:$test_feat" \
  2> $log
timer=$[$SECONDS-$timer]
echo "    execution time for generating results for test set = `utility/timer.pl $timer`"
cat $dir/test.mlf \
  | utility/result.htk2kaldi.pl \
  | python utility/word2char.py \
  > $dir/test.rec
cat $dir/test.rec \
  | python utility/compute-acc.py decode/test.text \
  > $dir/test.acc
acc=`grep "overall accuracy" $dir/test.acc | awk '{ print $4 }'`
echo "    result -> $dir/test.rec"
echo "    accuracy -> [ $acc ] %"

echo "Cleaning lattices of dev set generated during decoding process"
rm -f $dir/lat/*

sec=$SECONDS

echo ""
echo "Execution time for whole script = `utility/timer.pl $sec`"
echo ""

