# amharic-speech-lab

## Overview

**amharic-speech-lab** is a research-oriented personal project of mine focused on **automatic speech recognition (ASR)** and **text-to-speech (TTS)** for **Amharic**, a low-resource language that remains underrepresented in modern speech systems.

The project currently implements a clean, end-to-end baseline pipeline using Meta’s **Massively Multilingual Speech (MMS)** models:

- YouTube audio → WAV preprocessing
- Amharic ASR (speech → text)
- Amharic TTS (text → speech)

The goal of this repository is not only to demonstrate a working pipeline, but to serve as a foundation for future **fine-tuning, evaluation, and voice modeling experiments** for Amharic.

## Motivation

As kids, my brothers and I used to treat tools like Google Translate as a game of sorts. We grew up in Ethiopia, so Amharic was our mothertongue and English a close second. When we first started using google translate it was in earnest, and though it worked well in teaching us a few French phrases, we quickly figured out that its abilities were quite poor when it came to Amharic. So we started typing in the oddest Amharic words we would think of and then laughing at the strange, unintentionally funny outputs it would conjure up. It was extremely entertaining, but it also made clear how limited existing systems were when it came to the language. Alongside this, my interest in languages more broadly (shaped in part by studying Latin) and my background as a computer science major naturally pulled me toward machine learning as a way to think about language as a structured, evolving system.

Amharic is spoken by tens of millions of people, yet it remains low-resource in many modern speech models. This project is an initial attempt to explore that gap by building and evaluating strong multilingual ASR and TTS baselines, understanding where they break down, and iterating from there. It’s very much a work in progress (there are multiple rough edges and occasional mistakes that can be genuinely funny) so if nothing else, I hope anyone trying it out gets at least a small laugh from those moments, the same way my brothers and I did as kids.

## Current Capabilities

### YouTube Audio Ingestion

Audio is downloaded from YouTube and converted into a standardized format suitable for speech models:

- mono channel
- 16 kHz sample rate
- WAV (PCM)

This allows rapid testing on real-world Amharic speech.

### Automatic Speech Recognition (ASR)

- Model: `facebook/mms-1b-all` with Amharic adapter
- Architecture: CTC-based Wav2Vec2
- Output: Amharic text (Ethiopic script)

The ASR pipeline supports:
- resampling
- stereo → mono conversion
- optional formatting into multiple readable lines

Note: The model outputs lexical content only; punctuation and sentence boundaries are not predicted.

### Text-to-Speech (TTS)

- Model: `facebook/mms-tts-amh`
- Architecture: VITS
- Input: Amharic text (Ethiopic script)

Because MMS-TTS expects romanized input, the pipeline uses **uroman** as a preprocessing step. This behavior is documented in Meta’s MMS work, but it also highlights an important limitation: romanization can weaken prosody and naturalness.

The current TTS model is single-speaker. Voice characteristics (including perceived gender) are fixed by the training data and cannot be controlled at inference time.

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

Torch installation may vary by platform. If `pip install torch` fails, please follow the official PyTorch installation instructions for your system.

### Clone uroman

`uroman` is used for Ethiopic → Latin romanization during TTS preprocessing.

```bash
git clone https://github.com/isi-nlp/uroman.git uroman
```

## 🚀 Quickstart

### Download audio from YouTube

```bash
python scripts/yt_getwav.py "YOUTUBE_URL" -o soundoutput
```

### Run ASR

```bash
python scripts/asr_baseline.py \
  --in soundoutput/example.wav \
  --out textoutput/example.txt
```

### Run TTS

```bash
python scripts/tts_mms.py \
  --text-file textoutput/example.txt \
  --out soundoutput/example_tts.wav
```

## Known Limitations

- ASR output does not include punctuation or sentence boundaries
- TTS is single-speaker with no gender or style control
- Romanization is required for MMS-TTS, which may affect naturalness
- No fine-tuning has been applied yet

These limitations are just for the baseline project and the roadmap below shows some of my plans to address them.

## Roadmap

Planned next steps include:

- ASR fine-tuning
  - Convert the Kaggle Amharic Speech Corpus (Kaldi format) into Hugging Face datasets
  - Perhaps even create a new corpus that's extensive, it's low-resource for a reason and better organized data would definitely help
  - Fine-tune MMS ASR and evaluate using WER / CER

- Female-voice TTS fine-tuning
  - Curate female-speaker Amharic speech data
  - Fine-tune MMS-TTS to produce a distinct female voice
  - Compare intelligibility and naturalness against the baseline

- Evaluation & analysis
  - Quantitative ASR metrics
  - Qualitative TTS analysis (prosody, intelligibility)

## Acknowledgments

- Meta AI for the MMS models
- ISI NLP for `uroman`
- Kaggle contributors for open Amharic speech datasets

## License

This repository is intended for research and educational use. Please refer to individual model and dataset licenses for usage constraints.
