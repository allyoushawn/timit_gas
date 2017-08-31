#!/usr/bin/bash
. setup.sh

dir=exp/tri
feat_dev="cat $feat_loc/dev.39.cmvn.scp |"
compile-train-graphs $dir/tree $dir/final.mdl train/L.fst ark:train/dev.int ark:- |\
gmm-align-compiled exp/tri/final.mdl ark:- "scp,s,cs:$feat_dev"  ark:$dir/dev.ali
