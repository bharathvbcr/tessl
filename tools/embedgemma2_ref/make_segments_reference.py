#!/usr/bin/env python3
"""Mixed text-and-media items through the transformers processor and the model, for tessl's
segment-built items (tests/embedgemma2_segments_model.rs, python/tests/test_embedgemma2_segments.py).

    EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
      ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_segments_reference.py

Writes into target/embedgemma2_seg_ref/ (override: EMBEDGEMMA2_SEG_REF_DIR).

A case is a "message": text pieces and media in order. The snapshot's `chat_template.jinja` joins
the pieces with no separator and writes each medium as the processor's placeholder token
(`<|image|>`, `<|audio|>`, `<|video|>`); `EmbeddingGemma2Processor` then expands each placeholder
(`<boi> <image> x S <eoi>`, `<boa> <audio> x S <eoa>`, one `<boi> <video> x S <eoi>` block per
sampled frame, with `mm:ss` labels when the video processor's `add_timestamps` is on) and the
tokenizer wraps the string in `<bos>` and `<eos>`. This script renders that string itself and calls
the processor on it, so every case, the videos and the timestamped one included, takes one route.

For every case the directory holds

  {case}_ids.npy        the ids the processor made: what the library's own layout must reproduce
  {case}_emb_f32.npy    the embedding: the model's last hidden state, mean over every token, L2
                        normalized (the snapshot's 1_Pooling and 2_Normalize), fp32, sdpa, CPU

The pooling is done here, not by sentence-transformers, because its structured-message route does
not carry a video's frame rate (the sampler and the timestamps need it). The script checks its own
route against sentence-transformers' `encode` of the same message on the cases that route can take
(an image and a clip), and stops if they differ by more than 1e-6, so the reference is the
oracle's. The media are make_mm_reference.py's (same seeds):

  seg_image.png         make_image(224, 224, 1)
  seg_clip.wav          make_audio(1.0, 3) as 16 kHz mono PCM16 (the reference is given the array
                        the WAV holds)
  seg_video_raw.npy     make_video(24, 96, 176, seed 5), uint8 [T, H, W, 3] stored as int64 (the one
                        integer dtype tessl's npy reader takes), source fps 4.0

Cases: t_i_t, t_a_t, t_v_t, i_t, t_i_a_v, v_v (two videos in a row), t_vt_t (timestamped video).
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.io import wavfile
from sentence_transformers import SentenceTransformer
from transformers.video_utils import VideoMetadata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_mm_reference import SR, make_audio, make_image, make_video  # noqa: E402
from make_reference import provenance, snapshot  # noqa: E402

OUT = os.environ.get("EMBEDGEMMA2_SEG_REF_DIR", os.path.join(HERE, "..", "..", "target", "embedgemma2_seg_ref"))
VIDEO = (24, 96, 176, 4.0, 5)  # frames, height, width, source fps, seed (make_mm_reference.VIDEOS[0])

CASES = {
    "t_i_t": ["a photo of", "IMAGE", "on a table"],
    "t_a_t": ["listen:", "AUDIO", "(a chirp)"],
    "t_v_t": ["watch", "VIDEO", "now"],
    "i_t": ["IMAGE", "alone with text"],
    "t_i_a_v": ["x", "IMAGE", "y", "AUDIO", "z", "VIDEO"],
    "v_v": ["VIDEO", "VIDEO"],
    "t_vt_t": ["at", "VIDEO_T", "end"],
}
MEDIA = {"IMAGE", "AUDIO", "VIDEO", "VIDEO_T"}
# What the processor consumed that the model does not take (EmbeddingGemma2Processor.unused_input_names).
UNUSED = {"num_soft_tokens_per_image", "num_soft_tokens_per_video"}


def save(name, arr):
    np.save(os.path.join(OUT, name), arr)


def render(proc, pieces):
    """The chat template's string: text as is, each medium its placeholder, no separators."""
    token = {"IMAGE": proc.image_token, "AUDIO": proc.audio_token, "VIDEO": proc.video_token,
             "VIDEO_T": proc.video_token}
    return "".join(token.get(p, p) for p in pieces)


def embed(model, eg, proc, pieces, image, clip, frames, meta):
    text = render(proc, pieces)
    n_video = sum(p in ("VIDEO", "VIDEO_T") for p in pieces)
    proc.video_processor.add_timestamps = "VIDEO_T" in pieces
    kw = {}
    if "IMAGE" in pieces:
        kw["images"] = [[image] * pieces.count("IMAGE")]
    if "AUDIO" in pieces:
        kw["audio"] = [clip] * pieces.count("AUDIO")
        kw["sampling_rate"] = SR
    if n_video:
        kw["videos"] = [frames] * n_video
        kw["video_metadata"] = [meta] * n_video
    feats = proc(text=[text], return_tensors="pt", **kw)
    inputs = {k: v for k, v in feats.items() if k not in UNUSED}
    out = eg(**inputs)
    hidden = out.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1)
    emb = torch.nn.functional.normalize(pooled, dim=-1)[0]
    return inputs["input_ids"][0].cpu().numpy().astype(np.int64), emb


def st_message(pieces, image, clip):
    """The same message in sentence-transformers' structured form (images and clips only)."""
    content = []
    for p in pieces:
        if p == "IMAGE":
            content.append({"type": "image", "image": image})
        elif p == "AUDIO":
            content.append({"type": "audio", "audio": clip})
        else:
            content.append({"type": "text", "text": p})
    return [{"role": "user", "content": content}]


def main():
    path = snapshot()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    image = make_image(224, 224, 1)
    image.save(os.path.join(OUT, "seg_image.png"))
    clip0 = make_audio(1.0, 3)
    pcm = np.round(clip0 * 32767).astype(np.int16)
    wavfile.write(os.path.join(OUT, "seg_clip.wav"), SR, pcm)
    clip = pcm.astype(np.float32) / 32768.0
    t, h, w, fps, seed = VIDEO
    frames = make_video(t, h, w, seed)
    save("seg_video_raw.npy", frames.astype(np.int64))
    meta = VideoMetadata(total_num_frames=t, fps=fps, duration=t / fps)

    model = SentenceTransformer(path, device="cpu",
                                model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32})
    eg = [m for m in model.modules() if m.__class__.__name__ == "EmbeddingGemma2Model"][0]
    proc = model[0].processor
    record = {}
    with torch.no_grad():
        for name, pieces in CASES.items():
            ids, emb = embed(model, eg, proc, pieces, image, clip, frames, meta)
            if not any(p in ("VIDEO", "VIDEO_T") for p in pieces):
                via_st = model.encode([st_message(pieces, image, clip)], convert_to_tensor=True, batch_size=1)[0]
                err = float((via_st.float() - emb).abs().max())
                assert err < 1e-6, (name, "this script's route differs from sentence-transformers'", err)
                print(name, "sentence-transformers agrees", err)
            save(f"{name}_ids.npy", ids)
            save(f"{name}_emb_f32.npy", emb.detach().float().cpu().numpy().astype(np.float32))
            record[name] = {"pieces": pieces, "n_ids": int(ids.shape[0])}
            print(name, record[name])
    prov = provenance(path)
    prov["generator"] = "tools/embedgemma2_ref/make_segments_reference.py"
    with open(os.path.join(OUT, "cases.json"), "w") as f:
        json.dump({"snapshot": path, "video": {"frames": t, "height": h, "width": w, "fps": fps, "seed": seed},
                   "cases": record, "provenance": prov}, f, indent=1)
        f.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
