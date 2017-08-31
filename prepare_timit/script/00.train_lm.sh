dir=exp/lm

train_text=material/train.text
lexicon=material/lexicon.txt
lm_output=$dir/lm.arpa.txt

#srilm_bin=/share/tool
#srilm_bin=/usr/share/srilm/bin/i686-m64

mkdir -p $dir

# TODO:
#	1. process training text file
#	2. train a language model named $lm_output

ngram_order=1
kn_num=1

#cut -d ' ' -f 1 --complement $train_text > $dir/LM_train.txt
#ngram-count -order $ngram_order -kndiscount1 -text $dir/LM_train.txt \
#-vocab $lexicon -unk -lm $lm_output

cut -d ' ' -f 1  $lexicon > $dir/LM_train.txt
ngram-count -order $ngram_order -text $dir/LM_train.txt \
-vocab $lexicon -unk -lm $lm_output

