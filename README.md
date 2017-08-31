# Gate Activation Signals

This is the implementation of the paper *Gate Activation Signal Analysis for Gated Recurrent Neural Networks and Its Correlation with Phoneme Boundaries*. The paper is presented in **Interspeech 2017**. The slide of oral presentation in the conference is also included. Here is the [link](https://arxiv.org/abs/1703.07588) of the paper on arXiv.

Our corpus is **TIMIT**. We use **tensorflow 1.1.0** (with **CUDA 8.0**) with **python3.5** for our implementation.


### 1. Requirements
1. tensorflow 1.1.0 GPU-enabled version
2. numpy
3. sklearn
4. Kaldi
5. matplotlib (for plotting curves)

### 2. Run the code

Prepare data with your own timit corpus with **prepare_timit**:

1. Change directory to **prepare_timit**
2. Specify **timit_root**, **feat_loc** and **kaldi_root**
    in **setup.sh**
3. Execute **gen_data.sh** to extract training set and testing set MFCC features,
    the resulting scp files will be in
    **timit_gas/feature_scp, timit_gas/bounds**

The execute *./run.sh* to run the code.

### 3. Specify hyper-parameters

>run.sh: General settings
> * rnn_type: specify use gru or lstm
> * dropout_keep: specify dropout rate
> * gpu_device: specify gpu device id to be used
> * tol_window: specify the tolerance window size, 2 for 20 ms

>rnn_autoencoder.py: Training
> * scp_file: traiing data scp
> * dev_scp_file: development data scp
> * max_len: the max length of data for unrolling RNN. For TIMIT, the max len is 777 frames.
> * min_epoch: minimum epochs required for training

>decode.py: Testing
> * test_scp_file: testing data scp
> * test_bound_file: the bounds of testing data
> * max_len: the max length of data for unrolling RNN. For TIMIT, the max len is 777 frames.
> * enable_plt: if it is true, plot the curves of GAS and phoneme boundaries of a specified utterance.
> * th: thresholds for segmentation


### 4. GAS in code

 tensor name | GRU | LSTM
:-----------:|:---:|:-----:
 encoder_g1  | update gate | forget gate
 encoder_g2 | reset gate | input gate
 encoder_g3 | X | output gate

The encoder_g3 tensor for GRU will output all zeros.


### 5. Reference

Bibilographic inforamtion for this work:

```
@article{wang2017gate,
  title={Gate Activation Signal Analysis for Gated Recurrent Neural Networks and Its Correlation with Phoneme Boundaries},
  author={Wang, Yu-Hsuan and Chung, Cheng-Tao and Lee, Hung-yi},
  journal={arXiv preprint arXiv:1703.07588},
  year={2017}
}

```
