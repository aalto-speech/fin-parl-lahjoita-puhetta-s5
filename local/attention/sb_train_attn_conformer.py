#!/usr/bin/env python3
"""
Mostly taken from https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/transformer/train.py
"""

import os
import sys
import torch
import logging
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import LoopedLoader
from torch.utils.data import DataLoader
import unicodedata

import webdataset as wds
import glob
import io
import torchaudio
sys.path.append("local/")
import pathlib

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.wav
        tokens_bos, _ = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats)

        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps,) = predictions

        ids = batch.__key__
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Converted predicted tokens from indexes to words
                specials = [self.hparams.bos_index, self.hparams.eos_index, self.hparams.unk_index]
                hyps = [
                        [token for token in pred if token not in specials]
                        for pred in hyps
                ]
                predicted_words = [
                    self.hparams.tokenizer.decode_ids(prediction).split(" ")
                    for prediction in hyps
                ]
                target_words = [trn.split(" ") for trn in batch.trn]
                self.wer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        max_num_checkpoints=getattr(self.hparams, "average_n_ckpts", 1)
        if max_num_checkpoints > 1:
            ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key,
                min_key=min_key,
                max_num_checkpoints=max_num_checkpoints,
            )
            ckpt = sb.utils.checkpoints.average_checkpoints(
                ckpts, recoverable_name="model", device=self.device
            )

            self.hparams.model.load_state_dict(ckpt, strict=True)
            self.hparams.model.eval()
            print(f"Loaded the average of {len(ckpts)} best checkpoints")
        else:
            print(f"Loaded the checkpoint from {self.hparams.epoch_counter.current}")

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=10,
            )

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

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            #self.checkpointer.save_and_keep_only(
            #    meta={"ACC": 1.1, "epoch": epoch},
            #    max_keys=["ACC"],
            #    num_to_keep=1,
            #)

    def new_fit_dont_use(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """
        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ) and sb.utils.distributed.if_main_process():
            #HACK!
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            # HACK!
            if sb.utils.distributed.if_main_process():
                self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break


    def fit_batch(self, batch):

        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)
        else:
            if self.bfloat16_mix_prec:
                with torch.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_fit_start(self):
        super().on_fit_start()
        if self.optimizer_step > 0:
            self.optimizer.param_groups[0]['capturable'] = True


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

    def proper_sample(sample):
        if all(key in sample for key in ("transcript.txt", "audio.pth", "ali.pth")):
            return True
        else:
            return False
    traindata = (
            wds.WebDataset(hparams["trainshards"], nodesplitter=wds.split_by_node)
            .select(proper_sample)
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth", handler=wds.warn_and_continue)
            .map(tokenize)
            .repeat()
            .compose(
                hparams["train_dynamic_batcher"]
            )
    )
    # MEGAHACK! Crashes if there are not enough shards for every process.
    try:
        rank = int(os.environ["RANK"])
    except KeyError:
        #Not a distributed run
        rank=0
    validshards = glob.glob(f"./shards/dev_2k_{rank}/shard-000*.tar")
    def all_shards(src, group=None):
        for s in src:
            yield s
    if "valid_dynamic_batcher" in hparams:
        validdata = (
                wds.WebDataset(validshards, nodesplitter=all_shards)
                .decode()
                .rename(trn="transcript.txt", wav="audio.pth")
                .map(tokenize)
                .compose(
                    hparams["valid_dynamic_batcher"],
                )
        )
    else:
        raise ValueError("Need valid dynamic batcher")
    if "valid_subset" in hparams:
        validdata = validdata.slice(hparams["valid_subset"])


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
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets = dataio_prepare(hparams)

    # TODO: No pretraining atm

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    test_dataloader_opts = hparams.get("test_dataloader_opts", hparams["valid_dataloader_opts"])

    # These might be needed if webdataset is not detected
    #train_dataloader_opts.setdefault("batch_size", None)
    #valid_dataloader_opts.setdefault("batch_size", None)

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    #test_stats = asr_brain.evaluate(
    #    test_set=datasets[hparams["test_data_id"]],
    #    min_key="WER",
    #    test_loader_kwargs = test_dataloader_opts
    #)

