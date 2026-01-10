import argparse
from pathlib import Path

import torch
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ForCTC


def format_as_lines(text: str, words_per_line: int = 8) -> str:
    words = text.split()
    lines = []
    for i in range(0, len(words), words_per_line):
        lines.append(" ".join(words[i:i + words_per_line]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Amharic ASR using facebook/mms-1b-all + amh adapter"
    )
    parser.add_argument("--in", dest="audio_path", required=True, help="Path to input audio (wav/flac/mp3)")
    parser.add_argument("--out", dest="out_path", default="textoutput/asr.txt", help="Path to output text file")
    parser.add_argument(
        "--words-per-line",
        type=int,
        default=8,
        help="How many words to put on each line in the output (default: 8)",
    )
    args = parser.parse_args()

    audio_path = args.audio_path
    out_path = args.out_path

    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    processor.tokenizer.set_target_lang("amh")
    model.load_adapter("amh")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    waveform, sr = torchaudio.load(audio_path)

    # if it's stereo or multi-channel, average to mono
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    ids = torch.argmax(logits, dim=-1)[0]
    raw_text = processor.decode(ids).strip()
    text = format_as_lines(raw_text, words_per_line=args.words_per_line)

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path_p, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print("Wrote:", out_path_p)


if __name__ == "__main__":
    main()
