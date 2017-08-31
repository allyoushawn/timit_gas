#!/bin/bash

if [ -f setup.sh ]; then
  . setup.sh;
else
  echo "ERROR: setup.sh is missing!";
  exit 1;
fi

dir=exp/tree
srcdir=exp/mono
feat=$train_feat_setup

num_pdfs=2000
pdfs_per_group=5
num_groups=$[$num_pdfs/$pdfs_per_group]

mkdir -p $dir
mkdir -p $dir/log

echo "Senone clustering tree training :"
echo "    target -> [ $num_pdfs ] senones"

echo "    accumulating tree statistics"
if [ ! -f $dir/tree.acc ]; then
  log=$dir/log/tree.acc.log
  echo "        output -> $dir/tree.acc"
  echo "        log -> $log"
  silphonelist=`cat train/silphones.csl`
  acc-tree-stats --ci-phones=$silphonelist \
    $srcdir/final.mdl "ark,s,cs:$feat" ark:$srcdir/train.ali $dir/tree.acc \
    2> $log || exit 1
else
  echo "        $dir/tree.acc exists , skipping ..."
fi

echo "    computing questions for tree clustering"
if [ ! -f $dir/roots.txt ]; then
  echo "        output -> $dir/questions.qst $dir/roots.txt"
  utility/sym2int.pl train/phones.txt train/phonecluster.txt > $dir/phonesets.txt
  cluster-phones --pdf-class-list=0 $dir/tree.acc $dir/phonesets.txt $dir/questions.txt \
    2> $dir/log/cluster.log || exit 1
  compile-questions train/topo $dir/questions.txt $dir/questions.qst \
    2> $dir/log/questions.log || exit 1
  utility/sym2int.pl --ignore-oov train/phones.txt train/roots.txt > $dir/roots.txt
else
  echo "        $dir/roots.txt exists , skipping ..."
fi

echo "    building tree of two levels"
if [ ! -f $dir/tree ]; then
  log=$dir/log/train.tree.log
  echo "        output -> $dir/tree"
  echo "        log -> $log"
  build-tree-two-level --binary=false --verbose=1 --max-leaves-first=$num_groups \
    --max-leaves-second=$num_pdfs $dir/tree.acc $dir/roots.txt \
    $dir/questions.qst train/topo $dir/tree $dir/pdf2group.map \
    2> $log || exit 1
else
  echo "        $dir/tree exists , skipping ..."
fi

echo "Cleaning redundant materials generated during training process"
rm -f $dir/roots.txt
rm -f $dir/question.qst
rm -f $dir/phonesets.txt

sec=$SECONDS

echo ""
echo "Execution time for whole script = `utility/timer.pl $sec`"
echo ""

