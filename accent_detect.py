import sys, random, shutil, tempfile, subprocess, pathlib
from typing import List
from collections import Counter

import yt_dlp                         # pip install yt-dlp
from pydub import AudioSegment        # pip install pydub  (needs FFmpeg)
from speechbrain.pretrained.interfaces import foreign_class
import torch
# --------------------------------------------------------------------
# 1.  Config – where large model files live
# --------------------------------------------------------------------
MODEL_ID   = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"
MODEL_DIR  = pathlib.Path.home() / ".cache" / "speechbrain_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)    # ~/.cache/speechbrain_models

# --------------------------------------------------------------------
# 2.  Download video & extract WAV
# --------------------------------------------------------------------
def extract_audio(url: str, wav_path: pathlib.Path, sr: str = "44100") -> None:
    """Download <url> and convert its best audio stream to a WAV file."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg not found – install it with Homebrew or add to PATH")

    with tempfile.TemporaryDirectory() as tmp:
        ydl_opts = {
            "format": "bestaudio/best",
            "paths": {"home": tmp},
            "quiet": True,
            "noplaylist": True,            # <-- ignore playlist parts of the URL
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info    = ydl.extract_info(url, download=True)
            in_file = pathlib.Path(ydl.prepare_filename(info))

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(in_file),
             "-vn", "-acodec", "pcm_s16le", "-ar", sr, str(wav_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

# --------------------------------------------------------------------
# 3.  Make five random 5-second snippets
# --------------------------------------------------------------------
def save_random_5s_snippets(
    wav_path: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    count: int = 5,
    seed: int = 42
) -> List[pathlib.Path]:
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio   = AudioSegment.from_wav(wav_path)
    SEG_MS  = 5_000

    if len(audio) <= SEG_MS:                    # source shorter than 5 s
        out_file = out_dir / f"{wav_path.stem}_full.wav"
        audio.export(out_file, format="wav")
        return [out_file]

    paths = []
    for idx in range(1, count + 1):
        start = random.randint(0, len(audio) - SEG_MS)
        clip  = audio[start : start + SEG_MS]
        out   = out_dir / f"{wav_path.stem}_snippet{idx}.wav"
        clip.export(out, format="wav")
        paths.append(out)
    return paths

# --------------------------------------------------------------------
# 4.  Classify each snippet and print the probabilities & label
# --------------------------------------------------------------------
def classify_snippets(paths: List[pathlib.Path]) -> None:
    classifier = foreign_class(
        source=MODEL_ID,
        savedir=str(MODEL_DIR),
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
    )

    pooled_logits = None
    votes = []                           # ⬅︎ NEW

    for idx, wav in enumerate(paths, 1):
        out_prob, score, _, text_lab = classifier.classify_file(str(wav))
        label = text_lab[0] if isinstance(text_lab, (list, tuple)) else str(text_lab)
        score_val = float(score.item() if hasattr(score, "item") else score)

        print(f"[Snippet {idx}] {label:>10}   score={score_val:.4f}")

        votes.append(label)              # ⬅︎ NEW
        log_probs = torch.log(out_prob + 1e-8)
        pooled_logits = log_probs.clone() if pooled_logits is None else pooled_logits + log_probs

    pooled_logits = pooled_logits.squeeze(0)
    final_probs   = torch.softmax(pooled_logits, dim=-1)
    best_p, best_idx  = final_probs.max(dim=-1)
    final_label  = classifier.hparams.label_encoder.decode_torch(best_idx.unsqueeze(0))[0]

    # ⬇︎ NEW – majority percentage
    vote_share = 100 * Counter(votes)[final_label] / len(votes)

    print(f"\n>>> Clip-level prediction: {final_label}   "
          f"(p = {best_p.item():.3f}, vote share = {vote_share:.0f} %)")

# --------------------------------------------------------------------
# 5.  Glue everything together
# --------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python accent_detect.py <video_or_audio_url>")

    url       = sys.argv[1]
    data_dir  = pathlib.Path("data");     data_dir.mkdir(exist_ok=True)
    snip_dir  = pathlib.Path("snippets"); snip_dir.mkdir(exist_ok=True)

    wav_file  = data_dir / "download.wav"
    print("↓  Downloading & extracting audio …")
    extract_audio(url, wav_file)

    print("✂︎  Creating snippets …")
    snippets = save_random_5s_snippets(wav_file, snip_dir)

    print("🔍  Classifying accents …")
    classify_snippets(snippets)

if __name__ == "__main__":
    main()
