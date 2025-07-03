import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from encoder import audio
from encoder.config import librispeech_datasets, anglophone_nationalites
from encoder.params_data import *

_AUDIO_EXTENSIONS = ("wav", "flac", "m4a", "mp3")

class DatasetLog:
    """Creates and manages a log file that records dataset metadata."""
    def __init__(self, root, name):
        log_filename = "Log_%s.txt" % name.replace("/", "_")
        self.text_file = open(Path(root, log_filename), "w")
        self.sample_data = {}

        current_time = datetime.now().strftime("%A %d %B %Y at %H:%M")
        self.write_line("Dataset '%s' created on %s" % (name, current_time))
        self.write_line("-----")
        self._record_params()

    def _record_params(self):
        from encoder import params_data
        self.write_line("Current parameter settings:")
        for param in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param)
            self.write_line("\t%s: %s" % (param, value))
        self.write_line("-----")

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def add_sample(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.sample_data:
                self.sample_data[key] = []
            self.sample_data[key].append(value)

    def finalize(self):
        self.write_line("Summary statistics:")
        for key, values in self.sample_data.items():
            self.write_line("\t%s:" % key)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        finish_time = datetime.now().strftime("%A %d %B %Y at %H:%M")
        self.write_line("Completed on %s" % finish_time)
        self.text_file.close()


def _init_preprocess_dataset(dataset_name, datasets_root, out_dir) -> (Path, DatasetLog):
    dataset_root = datasets_root.joinpath(dataset_name)
    if not dataset_root.exists():
        print("Dataset directory %s not found, skipping." % dataset_root)
        return None, None
    return dataset_root, DatasetLog(out_dir, dataset_name)


def _preprocess_speaker(speaker_dir: Path, datasets_root: Path, out_dir: Path, skip_existing: bool):
    # Construct a unique speaker name based on its relative location
    speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)
    speaker_out_dir = out_dir.joinpath(speaker_name)
    speaker_out_dir.mkdir(exist_ok=True)
    sources_fpath = speaker_out_dir.joinpath("_sources.txt")

    if sources_fpath.exists():
        try:
            with sources_fpath.open("r") as f:
                existing_fnames = {line.split(",")[0] for line in f}
        except Exception:
            existing_fnames = {}
    else:
        existing_fnames = {}

    # Open the source file for appending or writing based on the skip_existing flag
    mode = "a" if skip_existing else "w"
    sources_file = sources_fpath.open(mode)
    audio_durations = []
    for ext in _AUDIO_EXTENSIONS:
        for in_fpath in speaker_dir.glob("**/*.%s" % ext):
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % ext, ".npy")
            if skip_existing and out_fname in existing_fnames:
                continue

            wav = audio.preprocess_wav(in_fpath)
            if len(wav) == 0:
                continue

            frames = audio.wav_to_mel_spectrogram(wav)
            if len(frames) < partials_n_frames:
                continue

            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))
            audio_durations.append(len(wav) / sampling_rate)

    sources_file.close()
    return audio_durations


def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger):
    print("%s: Processing %d speakers." % (dataset_name, len(speaker_dirs)))
    work_fn = partial(_preprocess_speaker, datasets_root=datasets_root, out_dir=out_dir, skip_existing=skip_existing)
    with Pool(4) as pool:
        tasks = pool.imap(work_fn, speaker_dirs)
        for durations in tqdm(tasks, dataset_name, len(speaker_dirs), unit="speakers"):
            for duration in durations:
                logger.add_sample(duration=duration)
    logger.finalize()
    print("Finished processing %s.\n" % dataset_name)


def preprocess_librispeech(datasets_root: Path, out_dir: Path, skip_existing=False):
    for dataset_name in librispeech_datasets["train"]["other"]:
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)


def preprocess_voxceleb1(datasets_root: Path, out_dir: Path, skip_existing=False):
    dataset_name = "VoxCeleb1"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return
    with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
        metadata = [line.split("\t") for line in metafile][1:]
    nationalities = {line[0]: line[3] for line in metadata}
    valid_speaker_ids = [sid for sid, nat in nationalities.items() if nat.lower() in anglophone_nationalites]
    print("VoxCeleb1: Using %d anglophone speakers out of %d." % (len(valid_speaker_ids), len(nationalities)))
    speaker_dirs = dataset_root.joinpath("wav").glob("*")
    speaker_dirs = [d for d in speaker_dirs if d.name in valid_speaker_ids]
    print("VoxCeleb1: Found %d anglophone speakers, %d missing (expected)." %
          (len(speaker_dirs), len(valid_speaker_ids) - len(speaker_dirs)))
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)


def preprocess_voxceleb2(datasets_root: Path, out_dir: Path, skip_existing=False):
    dataset_name = "VoxCeleb2"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return
    speaker_dirs = list(dataset_root.joinpath("dev", "aac").glob("*"))
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)
