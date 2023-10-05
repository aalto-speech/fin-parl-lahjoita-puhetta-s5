#!/usr/bin/env python3

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
sys.path.append("local/chain")
import pathlib

from make_shards import wavscp_to_output, segments_to_output, text_to_output, sync_streams, make_data_point

logger = logging.getLogger(__name__)


class KaldiData(torch.utils.data.IterableDataset):
    def __init__(self, path, tokenizer, bos_index, eos_index):
        self.testdir = pathlib.Path(path)
        self.tokenizer = tokenizer
        self.bos_index = bos_index
        self.eos_index = eos_index
        self._iterable = None

    @staticmethod
    def count_scp_lines(scpfile):
        lines = 0
        with open(scpfile) as fin:
            for _ in fin:
                lines += 1
        return lines

    def rename(self, sample):
        sample["trn"] = sample["transcript.txt"]
        sample["wav"] = sample["audio.pth"]
        del sample["transcript.txt"]
        del sample["audio.pth"]
        return sample

    def tokenize(self, sample):
        text = sample["trn"]
        # quick hack for one sample in text of test2021:
        text = text.replace(" <NOISE>", "")
        fulltokens = torch.LongTensor(
                [self.bos_index] + self.tokenizer.encode(text) + [self.eos_index]
        )
        sample["tokens"] = fulltokens[1:-1]
        sample["tokens_bos"] = fulltokens[:-1]
        sample["tokens_eos"] = fulltokens[1:]
        return sample

    def __len__(self):
        if (self.testdir / "segments").exists():
            return self.count_scp_lines(self.testdir / "segments")
        else:
            return self.count_scp_lines(self.testdir / "wav.scp")

    def __iter__(self):
        self._iterable = self._get_iterable()
        return self

    def __next__(self):
        outputs = next(self._iterable)
        datapoint = make_data_point(outputs)
        datapoint = self.rename(datapoint)
        datapoint = self.tokenize(datapoint)
        return datapoint
   
    def _get_iterable(self):
        if (self.testdir / "segments").exists():
            audio_iter = segments_to_output(self.testdir / "segments", self.testdir / "wav.scp")
        else:
            audio_iter = wavscp_to_output(self.testdir / "wav.scp")
        text_iter = text_to_output(self.testdir / "text")
        return sync_streams([audio_iter, text_iter], maxskip=0)


# Brain class for speech recognition training
class ShallowFusion(sb.Brain):
    def compute_forward(self, batch, stage):
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        feats = self.hparams.compute_features(batch.wav.data)
        feats = self.modules.normalize(feats, batch.wav.lengths)

        # Running the encoder (prevent propagation to feature extraction)
        encoded_signal = self.modules.encoder(feats.detach())
        
        predictions, _ = self.hparams.test_search(
            encoded_signal, batch.wav.lengths
        )

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        specials = [self.hparams.bos_index, self.hparams.eos_index, self.hparams.unk_index]
        predictions = [
                [token for token in pred if token not in specials]
                for pred in predictions
        ]
        predicted_words = [
            self.hparams.tokenizer.decode_ids(prediction).split(" ")
            for prediction in predictions
        ]
        target_words = [words.split(" ") for words in batch.trn]

        # Monitor word error rate and character error rated at
        # valid and test time.
        self.wer_metric.append(batch.__key__, predicted_words, target_words)
        self.cer_metric.append(batch.__key__, predicted_words, target_words)

        return torch.tensor([0.])

    def on_stage_start(self, stage, epoch):
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
        stage_stats = {}
        stage_stats["CER"] = self.cer_metric.summarize("error_rate")
        stage_stats["WER"] = self.wer_metric.summarize("error_rate")
        with open(self.hparams.wer_file, "w") as w:
            self.wer_metric.write_stats(w)
        with open(self.hparams.decode_text_file, "w") as fo:
            for utt_details in self.wer_metric.scores:
                print(utt_details["key"], " ".join(utt_details["hyp_tokens"]), file=fo)

    def on_evaluate_start(self, max_key=None, min_key=None):
        lm_ckpt = self.hparams.lm_ckpt_finder.find_checkpoint(min_key="loss")
        self.hparams.lm_pretrainer.collect_files(lm_ckpt.path)
        self.hparams.lm_pretrainer.load_collected(self.device)
        asr_ckpt = self.hparams.asr_ckpt_finder.find_checkpoint(min_key="WER")
        self.hparams.asr_pretrainer.collect_files(asr_ckpt.path)
        self.hparams.asr_pretrainer.load_collected(self.device)

    

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


    # Trainer initialization
    asr_brain = ShallowFusion(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
    )

    print("Decoding", hparams["test_data_id"])
    test_data = KaldiData(hparams["test_data_dir"], 
            tokenizer = hparams["tokenizer"],
            bos_index = hparams["bos_index"],
            eos_index = hparams["eos_index"])
    test_loader_kwargs = hparams.get("test_loader_kwargs", {})
    test_loader_kwargs.setdefault("batch_size", hparams.get("test_batch_size", 1))
    if "collate_fn" not in test_loader_kwargs:
        test_loader_kwargs["collate_fn"] = sb.dataio.batch.PaddedBatch

    test_stats = asr_brain.evaluate(
        test_set=test_data,
        min_key="WER",
        test_loader_kwargs = test_loader_kwargs
    )

