#!/usr/bin/env python3
"""Media files and sentence-transformers' embeddings of them, for tessl's native
media decoding (src/media/, tests/media_e2e_model.rs).

    EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
      ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_media_reference.py

Writes into target/media_e2e_ref/ (override: EMBEDGEMMA2_MEDIA_REF_DIR) the files
below and `{name}.emb_f32.npy` for each, from sentence-transformers 6.x /
transformers 5.x, fp32, sdpa, CPU, batch size 1.

  photo.png     224x224 RGB. Given to sentence-transformers as the path string:
                its processor then runs `torchvision.io.decode_image(path,
                mode=RGB)` (verified by tracing it; the PIL route with
                exif_transpose applies to PIL inputs only).
  photo.jpg     the same picture as JPEG (quality 90, 4:2:0). Same path route.
  clip16.wav    1.0 s, 16 kHz mono PCM16. sentence-transformers is given the
                array scipy reads (int16 / 32768), exactly what an audio loader
                returns: `transformers.load_audio` needs torchcodec or librosa,
                and neither is installed in this environment, so a .wav path
                cannot be given to it here.
  clip44.wav    the same clip as 44.1 kHz stereo PCM16 (the second channel
                half as loud), the array it is given being FFmpeg's decode to
                16 kHz mono with the channels averaged. FFmpeg's libswresample
                is the library torchcodec uses; it stands in for it here.
  clip44.m4a    the 44.1 kHz stereo clip as AAC, the array being FFmpeg's
                decode of that file to 16 kHz mono (channel mean).

The per-input explanation of what differs from tessl's own decode, and by how
much, is in docs/embedgemma2.md ("Media decoding").
"""

import os
import subprocess
import sys

import numpy as np
import torch
from scipy.io import wavfile
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_mm_reference import SR, make_audio, make_image  # noqa: E402
from make_reference import snapshot  # noqa: E402

OUT = os.environ.get("EMBEDGEMMA2_MEDIA_REF_DIR", os.path.join(HERE, "..", "..", "target", "media_e2e_ref"))
FFMPEG = "/opt/homebrew/bin/ffmpeg"


def ffmpeg_mono16k(path):
    """FFmpeg's decode of `path` to 16 kHz mono float32, channels averaged."""
    cmd = [FFMPEG, "-v", "error", "-i", path, "-af", "pan=mono|c0=0.5*c0+0.5*c1", "-ar", str(SR),
           "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, dtype="<f4").copy()


def main():
    os.makedirs(OUT, exist_ok=True)
    p = lambda n: os.path.join(OUT, n)  # noqa: E731
    img = make_image(224, 224, 1)
    img.save(p("photo.png"))
    img.save(p("photo.jpg"), quality=90, subsampling=2)
    clip = make_audio(1.0, 3)
    pcm = np.round(clip * 32767).astype(np.int16)
    wavfile.write(p("clip16.wav"), SR, pcm)
    t = np.arange(int(44100 * 1.0)) / 44100
    # A 44.1 kHz stereo rendering of the same sounds: the clip resampled by
    # FFmpeg for the left channel, the right at half the level.
    left = np.interp(t, np.arange(len(clip)) / SR, clip).astype(np.float32)
    stereo = np.stack([left, 0.5 * left], axis=1)
    wavfile.write(p("clip44.wav"), 44100, np.round(stereo * 32767).astype(np.int16))
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", p("clip44.wav"), "-c:a", "aac", "-b:a", "128k", p("clip44.m4a")],
                   check=True)

    cases = [
        ("photo.png", p("photo.png")),
        ("photo.jpg", p("photo.jpg")),
        ("clip16.wav", {"array": pcm.astype(np.float32) / 32768.0, "sampling_rate": SR}),
        ("clip44.wav", {"array": ffmpeg_mono16k(p("clip44.wav")), "sampling_rate": SR}),
        ("clip44.m4a", {"array": ffmpeg_mono16k(p("clip44.m4a")), "sampling_rate": SR}),
    ]
    model = SentenceTransformer(snapshot(), device="cpu",
                                model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32})
    with torch.no_grad():
        for name, item in cases:
            emb = model.encode([item], convert_to_tensor=True, batch_size=1)[0]
            np.save(p(f"{name}.emb_f32.npy"), emb.detach().float().cpu().numpy().astype(np.float32))
            print(name, tuple(emb.shape))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
