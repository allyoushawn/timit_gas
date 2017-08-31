#!/bin/bash

if [ -f setup.sh ]; then 
  . setup.sh;
else 
  echo "ERROR: setup.sh is missing!";
  exit 1; 
fi

#


#Config
phoneset=material/phoneset.txt
phonemap=material/phonemap.txt
roots=material/roots.txt
proto=material/topo.proto
spk2utt=material/spk2utt.txt
utt2spk=material/utt2spk.txt

#lexicons
lexicon_train=material/lexicon.txt
lexicon_decode=material/lexicon.txt

# TODO: try to replace with your language model
lm=material/lm.arpa.txt
#lm=exp/lm/lm.arpa.txt



train_text=material/train.text
dev_text=material/dev.text
test_text=material/test.text

#
echo "Data : "
echo "    phone set = $phoneset"
echo "    phone map = $phonemap"
echo "    phone tree root = $roots"
echo "    lexicon for training = $lexicon_train"
echo "    lexicon for decoding = $lexicon_decode"
echo "    language model for decoding = $lm"
echo "    phone HMM prototype = $proto"
echo "    training set label (text) = $train_text"
echo "    developing set label (text) = $dev_text"
echo "    testing set label (text) = $test_text"
echo ""

#

mkdir -p train
mkdir -p decode

# phone set operation

cat $phoneset | awk 'BEGIN{ print "<eps> 0"; } { printf("%s %d\n", $1, NR); }' \
  > train/phones.txt

ln -sf ../train/phones.txt decode/phones.txt

ln -sf ../$phonemap train/phonemap.txt

utility/silphones.pl train/phones.txt sil \
  train/silphones.csl train/nonsilphones.csl

cat material/phoneset.txt | utility/remove.silence.pl sil \
  > train/phonecluster.txt

ln -sf ../$roots train/roots.txt

# training lexicon operation

cat $lexicon_train | grep -v "<s>\|</s>" \
  > train/lexicon.txt

cat train/lexicon.txt | awk '{print $1}' | sort | uniq \
  | awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > train/words.txt

utility/make_lexicon_fst.pl train/lexicon.txt 0.5 sil \
  | fstcompile --isymbols=train/phones.txt \
    --osymbols=train/words.txt --keep_isymbols=false \
    --keep_osymbols=false \
  | fstarcsort --sort_type=olabel > train/L.fst

# phone model topology operation

silphonelist=`cat train/silphones.csl | sed 's/:/ /g'`
nonsilphonelist=`cat train/nonsilphones.csl | sed 's/:/ /g'`
sed -e "s:NONSILENCEPHONES:$nonsilphonelist:" \
  -e "s:SILENCEPHONES:$silphonelist:" $proto > train/topo

# label operation

cat $train_text | utility/sym2int.pl --ignore-first-field train/words.txt > train/train.int

cat $dev_text | utility/sym2int.pl --ignore-first-field train/words.txt > train/dev.int

# decoding material

ln -sf ../$lm decode/lm.arpa.txt

ln -sf ../$lexicon_decode decode/lexicon.txt

cat $dev_text | python utility/word2char.py > decode/dev.text
cat $test_text | python utility/word2char.py > decode/test.text

cat decode/dev.text | cut -d ' ' -f 1 > decode/dev.list
cat decode/test.text | cut -d ' ' -f 1 > decode/test.list

# decoding lexicon for WFST

cat decode/lexicon.txt \
  | awk '{print $1}' | sort | uniq \
  | awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > decode/words.txt

ndisambig=`utility/add_lex_disambig.pl decode/lexicon.txt decode/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1];

utility/add_disambig.pl --include-zero train/phones.txt $ndisambig \
  > decode/phones_disambig.txt

phone_disambig_symbol=`grep \#0 decode/phones_disambig.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 decode/words.txt | awk '{print $2}'`

( utility/make_lexicon_fst.pl decode/lexicon_disambig.txt 0.5 sil '#'$ndisambig \
    | fstcompile --isymbols=decode/phones_disambig.txt \
        --osymbols=decode/words.txt --keep_isymbols=false \
        --keep_osymbols=false \
    | fstaddselfloops  "echo $phone_disambig_symbol |" \
        "echo $word_disambig_symbol |" \
    | fstarcsort --sort_type=olabel ) 1> decode/L.fst 2> /dev/null

grep '#' decode/phones_disambig.txt | awk '{print $2}' > decode/disambig_phones.list

# decoding language model for WFST

cat decode/lm.arpa.txt \
  | utility/find_arpa_oovs.pl decode/words.txt > decode/oovs.txt

( cat decode/lm.arpa.txt \
    | egrep -v '<s> <s>|</s> <s>|</s> </s>' \
    | arpa2fst - | fstprint \
    | utility/eps2disambig.pl | utility/s2eps.pl \
    | utility/remove_oovs.pl decode/oovs.txt \
    | fstcompile --isymbols=decode/words.txt --osymbols=decode/words.txt \
      --keep_isymbols=false --keep_osymbols=false \
    | fstrmepsilon ) 1> decode/G.fst 2> /dev/null

# decoding LG 

( fsttablecompose decode/L.fst decode/G.fst \
  | fstdeterminizestar --use-log=true \
  | fstminimizeencoded ) 1> decode/LG.fst 2> /dev/null 

# all operations are finished 

sec=$SECONDS

echo "Execution time for whole script = `utility/timer.pl $sec`"
echo ""
