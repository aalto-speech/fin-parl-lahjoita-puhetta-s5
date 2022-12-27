#!/usr/bin/env python3
from collections import namedtuple
import textgrid
import pathlib

Segment = namedtuple("Segment", ["uttid", "recid", "spkid", "start", "stop", "text"])

def produce_segments(textgrids, margin=0.2, rec2dur={}):
    min_segment_len = margin*3  # An arbitrary but probably decent filter
    for tgpath in textgrids:
        tg = textgrid.TextGrid.fromFile(tgpath)
        ongoing_start = 0.
        ongoing_stop = 0.
        ongoing_text = []
        onrec_index = 1
        spkid = (pathlib.Path(tgpath).stem).split("_")[0]
        recid = pathlib.Path(tgpath).stem
        for intervaltier in tg.getList("words"):
            for interval in intervaltier:
                token = interval.mark
                start = interval.minTime
                stop = interval.maxTime
                if not token:
                    # No token is like silence
                    if ongoing_text:
                        uttid = recid + "-{:03d}".format(onrec_index)
                        onrec_index += 1
                        ongoing_stop = min(ongoing_stop+margin/2,
                                min(float(start)+margin/2, stop))
                        if ongoing_stop - ongoing_start > min_segment_len:
                            yield Segment(uttid=uttid,
                                    recid=recid,
                                    spkid=spkid,
                                    start=ongoing_start,
                                    stop=ongoing_stop,
                                    text=ongoing_text)
                    ongoing_start = max(stop-margin/2, ongoing_stop)
                    ongoing_text = []
                elif not ongoing_text:
                    ongoing_text.append(token)
                    ongoing_stop = stop
                    ongoing_start = max(ongoing_start, start-margin/2)
                else:
                    ongoing_text.append(token)
                    ongoing_stop = stop
            if ongoing_text:
                uttid = recid + "-{:03d}".format(onrec_index)
                if recid in rec2dur:
                    ongoing_stop = min(float(ongoing_stop)+margin/2, rec2dur[recid])
                if ongoing_stop - ongoing_start > min_segment_len:
                    yield Segment(uttid=uttid,
                            recid=recid,
                            spkid=spkid,
                            start=ongoing_start,
                            stop=ongoing_stop,
                            text=ongoing_text)


def overwrite_datadir(from_stream, datadir):
    datadir = pathlib.Path(datadir)
    datadir.mkdir(parents=True, exist_ok=True)
    with open(datadir / "segments", "w") as segments, \
         open(datadir / "text", "w") as text, \
         open(datadir / "utt2spk", "w") as utt2spk:
        for segment in from_stream:
            print(segment.uttid, segment.spkid, file=utt2spk)
            print(segment.uttid, " ".join(segment.text), file=text)
            print(segment.uttid, segment.recid, "{:.2f}".format(segment.start), "{:.2f}".format(segment.stop), file=segments)


if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("textgrids_glob")
    parser.add_argument("datadir_out")
    args = parser.parse_args() 
    textgrids = glob.glob(args.textgrids_glob)
    segment_stream = produce_segments(textgrids)
    overwrite_datadir(segment_stream, args.datadir_out)
