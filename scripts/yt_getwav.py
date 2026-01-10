import argparse
#to support command-line args like --mp3-quality
import shutil
#used  for shutil.which() to check if ffmpeg is installed
from pathlib import Path
#lets you build/handle filepaths cleanly and cross-platform
from yt_dlp import YoutubeDL
#imports yt-dlp's main python interface for downloading and post-processing


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        #checking id ffmpeg cand be found in your terminal PATH
        raise SystemExit("ffmpeg not found on PATH. Install ffmpeg first(brew/apt/winget) and try again."
        )


def download_audio_wav(url: str, out_dir: Path) -> list[Path]:
    "Download audio only and save it as a WAV (mono 16k) in out_dir."
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "ignoreerrors": True,

        "extract_flat": False, #to make sure we resolve full entries not flat urls
        "outtmpl": str(out_dir / "%(title).200s [%(id)s].%(ext)s"),

        # this makes it WAV instead of MP3 and also standardizes sample rate + channels
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "postprocessor_args": [
            "-ac", "1",      #mono
            "-ar", "16000",  #16k
        ],
    }

    wav_paths: list[Path] = []

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # yt-dlp returns playlist dicts differently depending on source
        if isinstance(info, dict) and info.get("_type") in ("playlist", "multi_video"):
            entries = info.get("entries") or []
            for entry in entries:
                if not entry:
                    continue
                base = Path(ydl.prepare_filename(entry))
                wav_path = base.with_suffix(".wav")
                if wav_path.exists():
                    wav_paths.append(wav_path)

        else:
            #single video
            base = Path(ydl.prepare_filename(info))
            wav_path = base.with_suffix(".wav")
            if wav_path.exists():
                wav_paths.append(wav_path)

    if not wav_paths:
        raise SystemExit(f"No WAV files were created in: {out_dir}")

    return wav_paths


def main():
    parser = argparse.ArgumentParser(
        description="Download audio from a YouTube video OR playlist and save WAV(s) into ./soundoutput"
    )
    parser.add_argument("url", help="YouTube video/playlist URL")

    parser.add_argument(
        "-o",
        "--out",
        default="soundoutput",
        help="Output folder (default: 'soundoutput')", #might raise quotation mark error
    )

    args = parser.parse_args()

    require_ffmpeg()

    out_dir = Path(args.out).expanduser().resolve()
    wav_paths = download_audio_wav(args.url, out_dir)

    print("\nDone:")
    for p in wav_paths:
        print(f"WAV: {p}")


if __name__ == "__main__":
    main()
