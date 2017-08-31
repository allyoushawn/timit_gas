#read -p "Are you sure you want to clean the directory? (Y/N) " sure
#if [ "$sure" == "Y" ]; then
  dir=(adapt decode  exp  train viterbi fst graph feat)
  set -x
  rm -rf ${dir[@]}
  rm -f log/*
#fi
