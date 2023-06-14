#!/usr/bin/env/python3
"""Finnish Parliament ASR
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import webdataset as wds
from glob import glob
import io
import torchaudio
sys.path.append("local/")
sys.path.append("local/attention")
import pathlib
from minwer_simple import minWER_loss_given
from speechbrain.utils.edit_distance import op_table, count_ops
import numpy as np
import unicodedata

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Runs all the computation of the CTC + seq2seq ASR. It returns the
        posterior probabilities of the CTC and seq2seq networks.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            At training time it returns predicted seq2seq log probabilities.
            If needed it also returns the ctc output log probabilities.
            At validation/test time, it returns the predicted tokens as well.
        """
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        feats, self.feat_lens = self.prepare_features(stage, batch.wav)
        tokens_bos, _ = self.prepare_tokens(stage, batch.tokens_bos)

        # Running the encoder (prevent propagation to feature extraction)
        encoded_signal = self.modules.encoder(feats.detach())

        # Embed tokens and pass tokens & encoded signal to decoder
        embedded_tokens = self.modules.embedding(tokens_bos)
        decoder_outputs, _ = self.modules.decoder(
            embedded_tokens, encoded_signal, self.feat_lens
        )

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(decoder_outputs)
        predictions = {"seq_logprobs": self.hparams.log_softmax(logits)}

        if self.is_ctc_active(stage) and stage == sb.Stage.TRAIN:
            # Output layer for ctc log-probabilities
            ctc_logits = self.modules.ctc_lin(encoded_signal)
            predictions["ctc_logprobs"] = self.hparams.log_softmax(ctc_logits)

        # MWER N-best
        # set max decoding step to the label length
        self.hparams.sampler.max_decode_ratio = (
            batch.tokens.data.size(1) / encoded_signal.size(1) * 1.5
        )
        (
            predicted_tokens,
            topk_scores,
            topk_hyps,
            topk_lens,
        ) = self.hparams.sampler(encoded_signal, self.feat_lens)
        predictions["p_tokens"] = predicted_tokens
        predictions["topk_scores"] = topk_scores
        predictions["topk_hyps"] = topk_hyps
        predictions["topk_lens"] = topk_lens
        #return p_seq, wav_lens, topk_hyps, topk_scores, topk_len
        if stage == sb.Stage.VALID:
            predictions["tokens"], _ = self.hparams.valid_search(
                encoded_signal, self.feat_lens
            )
        elif stage == sb.Stage.TEST:
            predictions["tokens"], _ = self.hparams.test_search(
                encoded_signal, self.feat_lens
            )

        return predictions

    def is_ctc_active(self, stage):
        """Check if CTC is currently active.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        """
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current
        return current_epoch <= self.hparams.number_of_ctc_epochs

    def prepare_features(self, stage, wavs):
        """Prepare features for computation on-the-fly

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        wavs : tuple
            The input signals (tensor) and their lengths (tensor).
        """
        wavs, wav_lens = wavs

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Feature computation and normalization
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        return feats, wav_lens

    def prepare_tokens(self, stage, tokens):
        """Double the tokens batch if features are doubled.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        tokens : tuple
            The tokens (tensor) and their lengths (tensor).
        """
        tokens, token_lens = tokens
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens], dim=0)
        return tokens, token_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs. We here
        do multi-task learning and the loss is a weighted sum of the ctc + seq2seq
        costs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        # Compute sequence loss against targets with EOS
        tokens_eos, tokens_eos_lens = self.prepare_tokens(
            stage, batch.tokens_eos
        )
        nll_loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["seq_logprobs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Add ctc loss if necessary. The total cost is a weighted sum of
        # ctc loss + seq2seq loss
        if self.is_ctc_active(stage):
            # Load tokens without EOS as CTC targets
            tokens, tokens_lens = self.prepare_tokens(stage, batch.tokens)
            loss_ctc = self.hparams.ctc_cost(
                predictions["ctc_logprobs"], tokens, self.feat_lens, tokens_lens
            )
            loss *= 1 - self.hparams.ctc_weight
            loss += self.hparams.ctc_weight * loss_ctc
        
        if getattr(self.hparams, "minwertype", "SubWER") == "SubWER":
            raise ValueError("Not supporting subword error rate any more")
        elif getattr(self.hparams, "minwertype", "SubWER") == "TrueWER":
            specials = [self.hparams.bos_index, self.hparams.eos_index, self.hparams.unk_index]
            batchsize = len(batch)
            wers = torch.zeros((batchsize,self.hparams.topk), dtype=torch.float32)
            for i, target in enumerate(batch.trn):
                # Ad hoc filter here:
                #target_words = [t in target.split() if t not in ["<UNK>"]]
                target_words = [t for t in target.split() if t not in ["<UNK>"]]
                for j, hyp in enumerate(predictions["topk_hyps"][i]):
                    hyp = hyp.cpu().tolist()
                    hyp = [token for token in hyp if token not in specials]
                    hyp = self.hparams.tokenizer.decode_ids(hyp).split(" ")
                    ops = op_table(target_words, hyp)
                    errors = sum(count_ops(ops).values())
                    wers[i,j] = errors
            minwerloss = minWER_loss_given(
                    wers = wers,
                    hypotheses_scores = predictions["topk_scores"],
                    subtract_avg = self.hparams.subtract_avg
            )
        loss = nll_loss * self.hparams.nll_weight + minwerloss

        if stage != sb.Stage.TRAIN:
            # Converted predicted tokens from indexes to words
            specials = [self.hparams.bos_index, self.hparams.eos_index, self.hparams.unk_index]
            predictions["tokens"] = [
                    [token for token in pred if token not in specials]
                    for pred in predictions["tokens"]
            ]
            predicted_words = [
                self.hparams.tokenizer.decode_ids(prediction).split(" ")
                for prediction in predictions["tokens"]
            ]
            target_words = [words.split(" ") for words in batch.trn]

            # Monitor word error rate and character error rated at
            # valid and test time.
            self.wer_metric.append(batch.__key__, predicted_words, target_words)
            self.cer_metric.append(batch.__key__, predicted_words, target_words)

        return loss

    def on_fit_start(self):
        super().on_fit_start()
        if self.optimizer_step > 0:
            self.optimizer.param_groups[0]['capturable'] = True

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
                num_to_keep=getattr(self.hparams, "ckpts_to_keep", 1)
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            if hasattr(self.hparams, "decode_text_file"):
                with open(self.hparams.decode_text_file, "w") as fo:
                    for utt_details in self.wer_metric.scores:
                        print(utt_details["key"], " ".join(utt_details["hyp_tokens"]), file=fo)

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        if getattr(self.hparams, "avg_ckpts", 1) > 1:
            ckpts = self.checkpointer.find_checkpoints(
                    max_key=max_key,
                    min_key=min_key,
                    max_num_checkpoints=self.hparams.avg_ckpts
            )
            model_state_dict = sb.utils.checkpoints.average_checkpoints(
                    ckpts, "model" 
            )
            self.hparams.model.load_state_dict(model_state_dict)
            self.checkpointer.save_checkpoint(name=f"AVERAGED-{self.hparams.avg_ckpts}")

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys mapping to 
        WebDataset datasets dataloaders for them.
    """
    translation_mapping = {
      "'": "",
      "-": "",
      "à": "a",
      "æ": "ä",
      "č": "c",
      "é": "e",
      "í": "i",
      "ñ": "nj",
      "ó": "o",
      "ø": "ö",
      "š": "sh",
      "ú": "u",
      "ü": "u",
      "ý": "y",
    }
    translation = str.maketrans(translation_mapping)

    def tokenize(sample, translation=translation):
        text = sample["trn"]
        # quick hack for one sample in text of test2021:
        text = text.replace("<UNK>", "")
        text = text.replace("[spn]", "")
        text = text.replace("[spk]", "")
        text = text.replace("[int]", "")
        text = text.replace("[fil]", "")
        text = text.replace(".fp", "")
        text = text.replace(".br", "")
        text = text.replace(".cough", "")
        text = text.replace(".ct", "")
        text = text.replace(".laugh", "")
        text = text.replace(".sigh", "")
        text = text.replace(".yawn", "")

        # Canonical forms of letters, see e.g. the Python docs
        # https://docs.python.org/3.7/library/unicodedata.html#unicodedata.normalize
        text = unicodedata.normalize("NFKC", text)
        # Just decide that everything will be uppercase:
        text = text.lower()
        # All whitespace to one space:
        text = " ".join(text.strip().split())
        # Translate weird chars:
        text = text.translate(translation)
        # Remove all extra characters:
        text = "".join(char for char in text if char.isalpha() or char == " " )
        text = text.replace(" <NOISE>", "")
        fulltokens = torch.LongTensor(
                [hparams["bos_index"]] + hparams["tokenizer"].encode(text) + [hparams["eos_index"]]
        )
        sample["tokens"] = fulltokens[1:-1]
        sample["tokens_bos"] = fulltokens[:-1]
        sample["tokens_eos"] = fulltokens[1:]
        return sample
    
    traindata = (
            wds.WebDataset(hparams["trainshards"])
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth")
            .map(tokenize)
            .repeat()
            .compose(
                hparams["train_dynamic_batcher"]
            )
    )
    if "valid_dynamic_batcher" in hparams:
        validdata = (
                wds.WebDataset(hparams["validshards"])
                .decode()
                .rename(trn="transcript.txt", wav="audio.pth")
                .map(tokenize)
                .compose(
                    hparams["valid_dynamic_batcher"],
                )
        )
    else:
        validdata = (
                wds.WebDataset(hparams["validshards"])
                .decode()
                .rename(trn="transcript.txt", wav="audio.pth")
                .map(tokenize)
                .batched(
                    batchsize=hparams["validbatchsize"], 
                    collation_fn=sb.dataio.batch.PaddedBatch,
                    partial=True
                )
        )

    normalizer = sb.dataio.preprocess.AudioNormalizer()
    def normalize_audio(sample):
        signal = sample["wav"]
        samplerate = sample["meta"]["samplerate"]
        sample["wav"] = normalizer(signal, samplerate)
        sample["meta"]["samplerate"] = normalizer.sample_rate 
        return sample

    datas = {"train": traindata, "valid": validdata, }
    
    return datas



if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # Pretrain if defined:
    if "pretrainer" in hparams:
        ckpt_finder_kwargs = hparams.get("ckpt_finder_kwargs", {"min_key": "WER"})
        ckpt = hparams["ckpt_finder"].find_checkpoint(**ckpt_finder_kwargs)
        hparams["pretrainer"].collect_files(ckpt.path)
        hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs = hparams["train_loader_kwargs"],
        valid_loader_kwargs = hparams.get("valid_loader_kwargs", {"batch_size": None})
    )

    # Load best checkpoint (highest STOI) for evaluation
    test_stats = asr_brain.evaluate(
        test_set=datasets[hparams["test_data_id"]],
        min_key="WER",
        test_loader_kwargs = hparams.get("test_loader_kwargs", {"batch_size": None})
    )
