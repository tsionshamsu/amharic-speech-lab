from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

MODEL_ID = "facebook/mms-tts-amh"


def uromanize(text: str, project_root: Path) -> str:
    """
    Romanize Ethiopic text to Latin using the local uroman/uroman/uroman.py script.
    Romanizing it works best for meta's model and they even mentioned this in their mms paper
    however I still want to find other models that might not need this step as it makes 
    the tts weaker.
    """
    uroman_repo = project_root / "uroman"

    # uroman has a couple directory layouts depending on version/packaging
    # so we check both and use whichever exists
    candidates = [
        (uroman_repo / "uroman.py", uroman_repo / "data"),
        (uroman_repo / "uroman" / "uroman.py", uroman_repo / "uroman" / "data"),
        (uroman_repo / "uroman" / "uroman.pl", uroman_repo / "uroman" / "data"),
    ]

    uroman_entry = None
    data_dir = None
    for entry, data in candidates:
        if entry.exists() and data.exists():
            uroman_entry = entry
            data_dir = data
            break

    if uroman_entry is None or data_dir is None:
        raise FileNotFoundError(
            "Could not find a usable uroman entrypoint + data directory.\n"
            f"Looked for:\n"
            f"  {uroman_repo}/uroman.py + {uroman_repo}/data\n"
            f"  {uroman_repo}/uroman/uroman.py + {uroman_repo}/uroman/data\n"
            f"  {uroman_repo}/uroman/uroman.pl + {uroman_repo}/uroman/data\n"
            f"Found uroman repo at: {uroman_repo}\n"
            "If you cloned uroman inside this repo, run `ls uroman` and `ls uroman/uroman` to confirm paths."
        )

    # important so relative data paths work
    cwd_dir = uroman_entry.parent.parent if uroman_entry.parent.name == "uroman" else uroman_repo

    # pick perl vs python runner depending on which file we found
    if uroman_entry.suffix == ".pl":
        cmd = ["perl", "-CSDA", str(uroman_entry)]
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd_dir),
        )
        stdout = proc.stdout.decode("utf-8", "ignore")
        stderr = proc.stderr.decode("utf-8", "ignore")
    else:
        cmd = [sys.executable, str(uroman_entry)]
        proc = subprocess.run(
            cmd,
            input=text,
            text=True,
            capture_output=True,
            cwd=str(cwd_dir),
        )
        stdout = proc.stdout
        stderr = proc.stderr

    if proc.returncode != 0:
        raise RuntimeError(
            "uroman failed\n"
            f"Exit code: {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{(stderr or '').strip()}\n"
        )

    out = (stdout or "").strip()
    if not out or out == "?":
        raise RuntimeError(
            "uroman output invalid (empty or '?')\n"
            f"Input:  {text!r}\n"
            f"Output: {out!r}\n"
            f"STDERR:\n{(stderr or '').strip()}\n"
        )

    return out


def synthesize_amharic(text_ethiopic: str, out_wav: Path, project_root: Path | None = None) -> None:
    if project_root is None:
        # lets this work whether you run it from repo root or from scripts/
        project_root = Path(__file__).resolve().parent.parent

    # Romanizing (as mentioned above, required for MMS-TTS)
    text_latn = uromanize(text_ethiopic, project_root)

    # Loading model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Tokenizing romanized text
    inputs = tokenizer(text_latn, return_tensors="pt")
    if inputs["input_ids"].shape[1] == 0:
        raise RuntimeError(
            "Tokenizer produced empty input_ids\n"
            f"Original:  {text_ethiopic!r}\n"
            f"Romanized: {text_latn!r}\n"
        )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generating waveform
    with torch.no_grad():
        waveform = model(**inputs).waveform

    audio = waveform[0].cpu().numpy()

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sr = getattr(model.config, "sampling_rate", 16000)
    sf.write(str(out_wav), audio, sr)

    print("Wrote    :", out_wav)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Amharic TTS using facebook/mms-tts-amh + local uroman"
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Amharic text in Ethiopic script",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="Path to a UTF-8 text file containing Amharic text (newlines are allowed)",
    )
    parser.add_argument(
        "--out",
        default="soundoutput/mms_amh_baseline.wav",
        help="Output wav path",
    )
    args = parser.parse_args()

    if args.text_file:
        p = Path(args.text_file)
        raw = p.read_text(encoding="utf-8")
        # join lines and normalize whitespace so TTS doesn't treat newlines weirdly
        text = " ".join(raw.split())
    elif args.text:
        text = args.text
    else:
        text = "ሰላም፣ እንዴት ነህ? ዛሬ ደስ ብሎኛል።"

    synthesize_amharic(text, Path(args.out))


if __name__ == "__main__":
    main()
