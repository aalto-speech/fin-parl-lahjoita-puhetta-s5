import torch
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.data_pipeline import takes, provides


@takes("tokens")
@provides("tokens_bos", "tokens_eos")
def to_eos_bos(tokens):
    yield tokens[:-1]
    yield tokens[1:]


def construct_static_lm_data(textfile, tokenizer, bos, eos):
    bosl = [bos]
    eosl = [eos]
    tokenized_sentences = {}
    with open(textfile) as fi:
        for lineno, line in enumerate(fi):
            text = line.strip()
            ids = tokenizer.encode_as_ids(text)
            tokenized_sentences[str(lineno)] = {"tokens": torch.LongTensor(bosl + ids + eosl)}
    dataset = DynamicItemDataset(tokenized_sentences)
    dataset.add_dynamic_item(to_eos_bos)
    dataset.set_output_keys(["id", "tokens_eos", "tokens_bos"])
    return dataset

class EndlessDynamicLMData(torch.utils.data.IterableDataset):
    def __init__(self, textfile, tokenizer, bos, eos, dynbatch_kwargs):
        super().__init__()
        self.textfile = textfile
        self.tokenizer = tokenizer
        self.bosl = [bos]
        self.eosl = [eos]
        self.dynbatch_kwargs = dynbatch_kwargs

    def __iter__(self):
        return sb.dataio.iterators.dynamic_bucketed_batch(
                self.repeated_data(),
                **self.dynbatch_kwargs
        )

    def repeated_data(self):
        while True:
            with open(self.textfile) as fi:
                for line in fi:
                    text = line.strip()
                    ids = self.tokenizer.encode_as_ids(text)
                    tokens = torch.LongTensor(self.bosl + ids + self.eosl)
                    yield {"tokens_bos": tokens[:-1],
                            "tokens_eos": tokens[1:]}

"""
# This versions adds bos and eos from tokenizer:
class EndlessDynamicLMData(torch.utils.data.IterableDataset):
    def __init__(self, textfile, tokenizer, dynbatch_kwargs):
        super().__init__()
        self.textfile = textfile
        self.tokenizer = tokenizer
        self.dynbatch_kwargs = dynbatch_kwargs

    def __iter__(self):
        return sb.dataio.iterators.dynamic_bucketed_batch(
                self.repeated_data(),
                **self.dynbatch_kwargs
        )

    def repeated_data(self):
        while True:
            with open(self.textfile) as fi:
                for line in fi:
                    text = line.strip()
                    ids = self.tokenizer.encode(text, add_bos=True, add_eos=True, out_type=int)
                    tokens = torch.LongTensor(ids)
                    yield {"tokens_bos": tokens[:-1],
                            "tokens_eos": tokens[1:]}
"""


def repeatedly(iterable):
    while True:
        for element in iterable:
            yield element


class EndlessDynBatchIterable(torch.utils.data.IterableDataset):
    def __init__(self, iterable, dynbatch_kwargs):
        self.iterable = sb.dataio.iterators.dynamic_bucketed_batch(
                repeatedly(iterable),
                **dynbatch_kwargs
        )

    def __iter__(self):
        return iter(self.iterable)

