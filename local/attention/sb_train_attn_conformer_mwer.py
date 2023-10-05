#!/usr/bin/env/python3
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

import webdataset as wds
import glob
import io
import torchaudio
sys.path.append("local/")
import pathlib
from minwer_simple import minWER_loss_given
from speechbrain.utils.edit_distance import op_table, count_ops
import numpy as np

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
        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)
        predictions = {"seq_logprobs":p_seq}

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
            # MWER N-best
            # set max decoding step to the label length
            self.hparams.sampler.max_decode_ratio = (
                batch.tokens.data.size(1) / enc_out.size(1) * 1.5
            )
            (
                predicted_tokens,
                topk_scores,
                topk_hyps,
                topk_lens,
            ) = self.hparams.sampler(enc_out.detach(), wav_lens)
            predictions["p_tokens"] = predicted_tokens
            predictions["topk_scores"] = topk_scores
            predictions["topk_hyps"] = topk_hyps
            predictions["topk_lens"] = topk_lens
            #return p_seq, wav_lens, topk_hyps, topk_scores, topk_len

        elif stage == sb.Stage.VALID:
            hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
        predictions["tokens"] = hyps

        return predictions

    def init_optimizers(self):
        #for modulename in getattr(self.hparams, "frozen_modules", []):
        #    getattr(self.modules, modulename).requires_grad_(False)
        #if self.distributed_launch:
        #    self.modules.CNN.module.requires_grad_(False)
        #    self.modules.Transformer.module.encoder.requires_grad_(False)
        #else:
        #    self.modules.CNN.requires_grad_(False)
        #    self.modules.Transformer.encoder.requires_grad_(False)
        super().init_optimizers()

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_seq = predictions["seq_logprobs"]
        hyps = predictions["tokens"] 

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


        if stage == sb.Stage.TRAIN:
            loss_seq = self.hparams.seq_cost(
                p_seq, tokens_eos, length=tokens_eos_lens
            ).sum()
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
            loss = loss_seq * self.hparams.nll_weight + minwerloss
        else:
            loss = torch.tensor([0.])

        if stage != sb.Stage.TRAIN:
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
                num_to_keep=20,
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
                #self.hparams.noam_annealing(self.optimizer)
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
                #self.hparams.noam_annealing(self.optimizer)

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()


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

    def tokenize(sample):
        text = sample["trn"]
        # quick hack for one sample in text of test2021:
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
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                **hparams["dynamic_batch_kwargs"]
            )
    )
    # MEGAHACK! Crashes if there are not enough shards for every process.
    try:
        rank = int(os.environ["RANK"])
    except KeyError:
        #Not a distributed run
        rank=0
    validshards = glob.glob(f"./shards/dev_all_{rank}/shard-000*.tar")
    if "valid_dynamic_batch_kwargs" in hparams:
        validdata = (
                wds.WebDataset(validshards)
                .decode()
                .rename(trn="transcript.txt", wav="audio.pth")
                .map(tokenize)
                .then(
                    sb.dataio.iterators.dynamic_bucketed_batch,
                    drop_end=False,
                    **hparams["valid_dynamic_batch_kwargs"]
                )
        )
    else:
        validdata = (
                wds.WebDataset(validshards)
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

    pt_kwargs = {}
    if hasattr(hparams, "test_max_key"):
        pt_kwargs["max_key"] = hparams.test_max_key
    elif hasattr(hparams, "test_min_key"):
        pt_kwargs["min_key"] = hparams.test_min_key

    if "device" in run_opts:
        device = run_opts["device"]
    elif "device" in hparams:
        device = hparams["device"]
    else:
        device = "cpu"
    pt_checkpointer = hparams["pretrain_checkpointer"]
    ckpt = pt_checkpointer.find_checkpoint(**pt_kwargs)
    print(f"Pretrain first from {ckpt.path}")
    pt_checkpointer.load_checkpoint(ckpt, device=device)
    if hparams.get("average_n_ckpts", 1) > 1:
        ckpts = pt_checkpointer.find_checkpoints(
            max_num_checkpoints=hparams["average_n_ckpts"],
            **pt_kwargs
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=device
        )
        hparams["model"].load_state_dict(ckpt, strict=True)
        print(f"Loaded the average of {len(ckpts)} best checkpoints")
    hparams["modules"]["CNN"].requires_grad_(False)
    hparams["modules"]["Transformer"].encoder.requires_grad_(False)
    
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

