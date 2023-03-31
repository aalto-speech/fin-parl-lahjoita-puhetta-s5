# Combined Finnish: Finnish Parliament and Lahjoita Puhetta

## Data prep

Look in run.sh and in local/ for the dataprep. This repository contains useful snippets for prepping Lahjoita Puhetta and Finnish Parliament Data alone, too.


## HMM systems

See run.sh

The basic HMM/GMM recipe outline is taken from Librispeech.
Then, DNN acoustic models are trained in SpeechBrain.

For an example of simple batch transcription, see transcribe-hmmdnn-basic.sh


## AED models

See run-attn.sh


## wav2vec 2.0

There are basic outlines for training wav2vec 2.0 models, copied from earlier experiments we did on Librispeech. However, the basic configuration did not produce good results, and we left the experiments there, as this data size is already very demanding, even without huge wav2vec 2.0 Encoders.

Further work would almost certainly find improvements with large SSL Transformer-based models though.

