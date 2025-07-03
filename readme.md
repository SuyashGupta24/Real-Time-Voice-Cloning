## 1. Install Requirements
 Both Windows and Linux are supported. A GPU is recommended for training and for
inference speed, but is not mandatory.
## 2. Python 3.7 is recommended. Python 3.5 or greater should work, but you'll probably have to
tweak the dependencies' versions. I recommend setting up a virtual environment using `venv`,
but this is optional.
## 3. Install ffmpeg. This is necessary for reading audio files.
## 4. Install PyTorch. Pick the latest stable version, your operating system, your package manager
(pip by default) and finally pick any of the proposed CUDA versions if you have a GPU,
otherwise pick CPU. Run the given command.
## 5. Install the remaining requirements with `pip install -r requirements.txt`

## 6. (Optional) Download Datasets
For playing with the toolbox alone, I only recommend downloading `LibriSpeech/train-clean-
100`. Extract the contents as `<datasets_root>/LibriSpeech/train-clean-100` where
`<datasets_root>` is a directory of your choosing. Other datasets are supported in the toolbox, see
here. You're free not to download any dataset, but then you will need your own data as audio files
or you will have to record it with the toolbox.
## 7. Launch the Toolbox
You can then try the toolbox:

python demo_toolbox.py



## OVERVIEW

# Voice Cloning - User-Friendly Edition

## Overview

Voice Cloning - User-Friendly Edition is a real-time voice cloning system designed for simplicity and high performance. It allows you to replicate voices with impressive quality through an intuitive interface and streamlined workflow.

## Features

- **Real-Time Cloning:** Generate voice replicas instantly.
- **User-Friendly Interface:** A simple design that makes setup and use effortless.
- **Optimized Performance:** Designed for fast and efficient processing in practical scenarios.

## Model Architecture

Inspired by state-of-the-art methods, the system employs a modular architecture that includes:
- **Speaker Encoder:** Captures and encodes distinctive voice features.
- **Synthesizer:** Converts textual input into a mel-spectrogram representation.
- **Vocoder:** Transforms mel-spectrograms into high-fidelity audio waveforms.

## Installation

1. **Unzip the Package:** Extract the contents to your desired directory.
2. **Install Dependencies:** Open your terminal in the project directory and run:
   ```bash
   pip install -r requirements.txt
