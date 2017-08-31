
if [ -f setup.sh ]; then
  . setup.sh;
else
  echo "ERROR: setup.sh is missing!";
  exit 1;
fi


dir=fst/mono
model=exp/mono/final.mdl
HCLG=graph/mono/HCLG.fst
dev_label=decode/dev.text
test_label=decode/test.text

dev_feat=$dev_feat_setup
test_feat=$test_feat_setup

dev_acwt=0.1
dev_beam=13.0
dev_lat_beam=6.0
test_beam=15.0

mkdir -p $dir

timer=$SECONDS
echo "Generating lattices for dev set"
gmm-latgen-faster --beam=$dev_beam --lattice-beam=$dev_lat_beam \
  --acoustic-scale=$dev_acwt --word-symbol-table=decode/words.txt \
  $model $HCLG "ark:$dev_feat" \
  ark,t:$dir/dev.lat 2> $dir/latgen.dev.log
timer=$[$SECONDS-$timer]
echo "Execution time for generating lattices = `utility/timer.pl $timer`"

timer=$SECONDS
max_acc=0.0
opt_acwt=none
echo "Recoring lattice to find optimal acoustic weight"
for x in `seq 1 10`; do
  acwt=`perl -e "printf (\"%4.2f\", $x * 0.02);"`
  echo "Rescoring lattice using acoustic weight = $acwt -> $dir/dev.$acwt.tra"
  lattice-best-path --acoustic-scale=$acwt --word-symbol-table=decode/words.txt \
    ark:$dir/dev.lat ark,t:$dir/dev.$acwt.tra ark,t:$dir/dev.$acwt.ali \
    2> $dir/rescore.$acwt.log
  echo "Generating char level transcription -> $dir/dev.$acwt.rec"
  cat $dir/dev.$acwt.tra \
    | utility/int2sym.pl --ignore-first-field decode/words.txt \
    | python utility/word2char.py \
    > $dir/dev.$acwt.rec
  cat $dir/dev.$acwt.rec \
    | python utility/compute-acc.py decode/dev.text \
    > $dir/dev.$acwt.acc
  acc=`grep "overall accuracy" $dir/dev.$acwt.acc | awk '{ print $4 }'`
  if [ "1" == `awk "BEGIN{print($acc>$max_acc)}"` ]; then
      opt_acwt=$acwt
      max_acc=$acc
  fi
  echo "    acoustic weight = [ $acwt ]"
  echo "        result -> $dir/dev.$acwt.rec"
  echo "        accuracy -> [ $acc ] %"
done
timer=$[$SECONDS-$timer]
echo "Execution time for rescoring lattices = `utility/timer.pl $timer`"
echo "Optimal acoustic weight = $opt_acwt"

timer=$SECONDS
echo "Recognizing test set with acoustic weight = $opt_acwt"
gmm-decode-faster --beam=$test_beam --acoustic-scale=$opt_acwt \
  --word-symbol-table=decode/words.txt $model \
  $HCLG "ark:$test_feat" ark,t:$dir/test.tra ark,t:$dir/test.ali 2> $dir/decode.test.log
timer=$[$SECONDS-$timer]
echo "Execution time for recognizing test set = `utility/timer.pl $timer`"

echo "Generating char level transcription -> $dir/test.rec"
cat $dir/test.tra \
  | utility/int2sym.pl --ignore-first-field decode/words.txt \
  | python utility/word2char.py \
  > $dir/test.rec
cat $dir/test.rec \
  | python utility/compute-acc.py decode/test.text \
  > $dir/test.acc
acc=`grep "overall accuracy" $dir/test.acc | awk '{ print $4 }'`
echo "    result -> $dir/test.rec"
echo "    accuracy -> [ $acc ] %"

sec=$SECONDS

echo ""
echo "Execution time for whole script = `utility/timer.pl $sec`"
echo ""

