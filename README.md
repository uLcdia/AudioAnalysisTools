# Audio Analysis Tools

## Overview

A Python toolkit for audio signal processing and analysis, covering waveform processing, spectrogram analysis, feature extraction, speech processing, and speaker verification.

## Setup

Clone the repository:
```bash
git clone https://github.com/uLcdia/AudioAnalysisTools
cd AudioAnalysisTools
```

Install dependencies:
```bash
uv sync
```

First read the usage below, then run each task with `uv run taskNum.py`

## Task 1: Waveform Processing and Visualization

Extracts, manipulates (fade-in/out, repeat), and visualizes audio waveforms using pydub and librosa. Compares original and processed waveforms.

### Usage

Place audio files in `samples/` (e.g., `sample1.wav`, `sample2.wav`).

## Task 2: Spectrogram Analysis

Generates and compares spectrograms using `scipy.signal.spectrogram` and `matplotlib.pyplot.specgram` to analyze frequency content in different environments.

### Usage

Uses the same audio files as Task 1.

## Task 3: Acoustic Feature Extraction

Extracts and visualizes MFCC and Filter Bank features using `python_speech_features` and matplotlib. Compares features across different recording conditions.

### Usage

Uses the same audio files as previous tasks.

## Task 4: Speech Recognition and Synthesis

Performs speech-to-text (Whisper) and text-to-speech (Coqui-TTS) with optional intelligent sentence segmentation (LLM) and voice cloning. Includes fallback mechanisms and API alternatives (Hugging Face).

### Features

- Multilingual speech recognition (Mandarin/dialect).
- LLM-based or fallback sentence segmentation.
- Multilingual speech synthesis with voice cloning or standard voices.
- GPU acceleration support.

### Usage

Add files like `sample3.wav`, `sample4.wav` to `samples/`. Requires Hugging Face API token in `.env` for some features.
Output is saved in `task4_output/`.

## Task 5: Speaker Verification

Verifies if two audio recordings are from the same speaker using SpeechBrain's ECAPA-TDNN model and cosine similarity, independent of language or content. Visualizes similarity matrix.

### Usage

Add files like `sample5.wav`, `sample6.wav`, and `sample.wav` (different speaker) to `samples/`.
