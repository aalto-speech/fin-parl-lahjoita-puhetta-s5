#!/usr/bin/env/python3
"""Finnish Parliament ASR
"""

import os
import os.path
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

from sb_train_attn_w2v2 import ASR, dataio_prepare
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

if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    print("Decoding", hparams["test_data_id"])
    test_data = KaldiData(hparams["test_data_dir"], 
            tokenizer = hparams["tokenizer"],
            bos_index = hparams["bos_index"],
            eos_index = hparams["eos_index"])
    test_loader_kwargs = hparams.get("test_loader_kwargs", {})
    if "collate_fn" not in test_loader_kwargs:
        test_loader_kwargs["collate_fn"] = sb.dataio.batch.PaddedBatch

    test_stats = asr_brain.evaluate(
        test_set=test_data,
        min_key="WER",
        test_loader_kwargs = test_loader_kwargs
    )
