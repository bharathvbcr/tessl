#!/usr/bin/env python3
"""Multimodal reference outputs for EmbeddingGemma 2's vision and audio towers.

    ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_mm_reference.py [SNAPSHOT_DIR]

The companion of make_reference.py (text). SNAPSHOT_DIR is
google/embeddinggemma-2; default: the Hugging Face cache. Writes into
target/embedgemma2_mm_ref/ (override with EMBEDGEMMA2_MM_REF_DIR).

The inputs are synthetic and deterministic (no downloaded media): two images
of different sizes and aspect ratios, two audio clips of different
lengths, and two videos (different frame counts, sizes and source frame
rates; the second has more sampled frames than the processor's 32-frame cap)
plus the first video again with `add_timestamps` on. Each runs alone through
sentence-transformers (batch size 1), fp32,
on the CPU, with sdpa attention: sentence-transformers' default, and the
implementation whose audio mask is right. EMBEDGEMMA2_MM_ATTN=eager selects
eager instead, which in transformers 5.19 inverts the audio tower's chunked
local-attention mask (create_bidirectional_mask returns a float additive mask
under eager, and Gemma4AudioAttention.forward fills mask.logical_not(), so
the visible keys are the ones blanked); tests/embedgemma2_audio_model.rs pins
that disagreement against an eager directory. Text and vision differ between
the two only by rounding. The processor's own outputs are saved as the
model received them, so tessl's towers can start from exactly those tensors; the
raw inputs are saved beside them, so tessl's own preprocessing
(src/embedgemma2_pre.rs) can be checked against the processor's outputs and
carried through the towers to the embedding.

For input n of kind k (image, audio or video), prefix `{k}{n}_` (`video0t_` is
video 0 with timestamps):

  ids.npy              the token ids the model received, placeholders included
  attention_mask.npy   the text attention mask
  raw.npy              the raw input the processor received: image [H, W, 3]
                       RGB bytes as int64, audio [N] float32 at 16 kHz (the input of
                       tessl's native preprocessing, src/embedgemma2_pre.rs)
  image:  pixel_values.npy [1, P, 768], image_position_ids.npy [1, P, 2]
          (x, y; padding patches are (-1, -1))
          patch_embed.npy  [P, 768]  patch embedder output (projection + 2D
                                     position embedding, before layer 0)
          layer{i}.npy     [P, 768]  vision encoder output after layer i
          pooled.npy       [S, 768]  the vision tower's output (k x k
                                     average pool by position, times
                                     sqrt(768), padding stripped)
  audio:  input_features.npy [1, F, 128], input_features_mask.npy [1, F]
          subsample.npy    [T, 1024] after the conv subsampling projection
          layer{i}.npy     [T, 1024] conformer output after layer i
          tower_out.npy    [T, 1536] after output_proj
          out_mask.npy     [T]       the tower's output mask (True = valid)
  video:  raw.npy [T, H, W, 3] RGB bytes as int64 (every decoded frame),
          fps.npy f64 scalar (the source frame rate; the metadata passed in
          is {fps, duration = T / fps, total_num_frames = T}, without which the
          video processor keeps every frame and skips fps sampling),
          indices.npy [F] int64 the sampled frames' indices in the source,
          timestamps.npy [F] f64 (index / fps),
          pixel_values_videos.npy [F, P, 768], video_position_ids.npy
          [F, P, 2] (the processor's tensors, frames concatenated, P = 1260),
          soft_per_frame.npy [1] int64; soft.npy is [F * S, 512], frame by
          frame
  soft.npy             [S, 512]  soft tokens after the multimodal embedder
                                 (weightless RMSNorm + projection), valid
                                 rows only, in the order they are scattered
  lm_final.npy         [L, 512]  the text model after its final norm
  emb_f32.npy          [768]     sentence-transformers' embedding (fp32)
  emb_bf16.npy         [768]     the same with the checkpoint's bf16 weights

inputs.json records each input's construction and shape; provenance.json the
library versions, snapshot and commit. Needs torch, numpy, pillow,
transformers >= 5.19 and sentence-transformers >= 6.1.
"""

import importlib.metadata
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_reference import provenance, snapshot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EMBEDGEMMA2_MM_REF_DIR", os.path.join(HERE, "..", "..", "target", "embedgemma2_mm_ref"))
SR = 16000
ATTN = os.environ.get("EMBEDGEMMA2_MM_ATTN", "sdpa")
if ATTN not in ("sdpa", "eager"):
    sys.exit(f"EMBEDGEMMA2_MM_ATTN must be sdpa or eager, not {ATTN!r}")


def make_image(w, h, seed):
    """A smooth gradient with a disc and a checker patch: low and high frequencies, every channel distinct."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 127.5 * (1 + np.sin(2 * math.pi * x / w))
    g = 255.0 * y / max(h - 1, 1)
    b = 127.5 * (1 + np.cos(2 * math.pi * (x + y) / (w + h)))
    img = np.stack([r, g, b], axis=-1)
    cx, cy, rad = 0.6 * w, 0.4 * h, 0.2 * min(w, h)
    disc = (x - cx) ** 2 + (y - cy) ** 2 < rad**2
    img[disc] = [240.0, 30.0, 60.0]
    checker = ((x // 8 + y // 8) % 2 == 0) & (x < w // 4) & (y > 3 * h // 4)
    img[checker] = [10.0, 10.0, 10.0]
    img += rng.normal(0, 4.0, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def make_audio(seconds, seed):
    """A chirp plus two steady tones and quiet noise, peak 0.5, at 16 kHz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    chirp = np.sin(2 * math.pi * (200 * t + 0.5 * (3000 - 200) / seconds * t**2))
    tones = 0.5 * np.sin(2 * math.pi * 440 * t) + 0.3 * np.sin(2 * math.pi * 1250 * t + 0.7)
    a = chirp + tones + 0.05 * rng.standard_normal(t.shape)
    return (0.5 * a / np.abs(a).max()).astype(np.float32)


def make_video(t, h, w, seed):
    """Frames that differ: a gradient photo with a moving disc and per-frame noise, [t, h, w, 3] uint8."""
    rng = np.random.default_rng(seed)
    base = np.asarray(make_image(w, h, seed), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    frames = []
    for f in range(t):
        fr = base.copy()
        cx = (3 * f) % w
        fr[(xx - cx) ** 2 + (yy - h // 2) ** 2 < (min(h, w) // 4) ** 2] = [250.0, 250.0, 20.0]
        fr += rng.normal(0, 3.0, fr.shape)
        frames.append(np.clip(fr, 0, 255).astype(np.uint8))
    return np.stack(frames)


IMAGES = [(224, 224, 1), (640, 360, 2)]
AUDIOS = [(1.0, 3), (3.3, 4)]
# (frames, height, width, source fps, seed). Sampling at the processor's 1 fps:
# video 0 is 6 s, so 6 frames (source indices 0, 4, ..., 20); video 1 is 40 s,
# so 40 sampled frames, which the 32-frame cap cuts to 32 (uniform).
VIDEOS = [(24, 96, 176, 4.0, 5), (100, 72, 104, 2.5, 6)]
PROCESSOR_FPS, MAX_FRAMES = 1, 32


def save(prefix, name, arr):
    np.save(os.path.join(OUT, f"{prefix}{name}.npy"), arr)


def f32(t):
    return t.detach().float().cpu().numpy().astype(np.float32)


def m32_eg(model):
    """The EmbeddingGemma2Model inside a SentenceTransformer."""
    return [m for m in model.modules() if m.__class__.__name__ == "EmbeddingGemma2Model"][0]


def run(model, item):
    """Encode one input; return the embedding and every captured intermediate."""
    eg = m32_eg(model)
    cap = {"layers": []}
    hooks = []

    # sentence-transformers calls `forward` by name (its modality config's
    # "method"), which skips module hooks on this module, so wrap it instead.
    inner = eg.forward

    def forward(*args, **kwargs):
        assert not args, "EmbeddingGemma2Model.forward takes keyword inputs here"
        cap["kwargs"] = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        return inner(**kwargs)

    eg.forward = forward
    vt, at, lm = eg.vision_tower, eg.audio_tower, eg.language_model

    def grab(key):
        def hook(_m, _i, out):
            cap[key] = out[0] if isinstance(out, tuple) else out
        return hook

    def grab_layer(_m, _i, out):
        cap["layers"].append(out[0] if isinstance(out, tuple) else out)

    hooks.append(vt.patch_embedder.register_forward_hook(grab("patch_embed")))
    hooks += [layer.register_forward_hook(grab_layer) for layer in vt.encoder.layers]
    hooks.append(vt.register_forward_hook(lambda _m, _i, out: cap.__setitem__("pooled", out.last_hidden_state)))
    hooks.append(eg.embed_vision.register_forward_hook(grab("soft_v")))
    hooks.append(at.subsample_conv_projection.register_forward_hook(grab("subsample")))
    hooks += [layer.register_forward_hook(grab_layer) for layer in at.layers]
    hooks.append(at.register_forward_hook(lambda _m, _i, out: cap.update(
        tower_out=out.last_hidden_state, out_mask=out.attention_mask)))
    hooks.append(eg.embed_audio.register_forward_hook(grab("soft_a")))
    hooks.append(lm.norm.register_forward_hook(grab("lm_final")))
    try:
        emb = model.encode([item], convert_to_tensor=True, batch_size=1)[0]
    finally:
        del eg.forward
        for h in hooks:
            h.remove()
    return emb, cap


def expected_indices(n_frames, fps):
    """The frames the video processor keeps, from its rules written out: `int(duration * 1)` frames at
    `int(i * fps)`, then `np.linspace(0, n - 1, 32, dtype=int)` of them if more than 32."""
    sampled = max(1, int((n_frames / fps) * PROCESSOR_FPS))
    idx = [min(n_frames - 1, int(i * fps / PROCESSOR_FPS)) for i in range(sampled)]
    if len(idx) > MAX_FRAMES:
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, MAX_FRAMES, dtype=int)]
    return idx


def set_timestamps(model, on):
    """`add_timestamps` of the processor's video processor (what `replace_video_token` reads)."""
    model[0].processor.video_processor.add_timestamps = bool(on)


def save_video(model, prefix, kw, cap, desc, ids, snapshot_dir):
    """A video's processor tensors, sampled indices and timestamps; checks the golden is the fps-sampled
    path and that ids have the layout the Rust builder reproduces. Returns what inputs.json records."""
    from transformers.video_utils import VideoMetadata

    n_frames = int(kw["num_frames_per_video"].sum())
    want = expected_indices(desc["frames"], desc["fps"])
    # The golden must be the fps-sampled path, not "every frame".
    assert n_frames == len(want), (prefix, n_frames, len(want))
    meta = VideoMetadata(total_num_frames=desc["frames"], fps=desc["fps"], duration=desc["frames"] / desc["fps"])
    idx = [int(i) for i in model[0].processor.video_processor.sample_frames(
        meta, fps=PROCESSOR_FPS, max_frames=MAX_FRAMES, overflow_strategy="uniform")]
    assert idx == want, (prefix, idx, want)
    save(prefix, "fps", np.array(desc["fps"], dtype=np.float64))
    save(prefix, "indices", np.array(idx, dtype=np.int64))
    save(prefix, "timestamps", np.array(idx, dtype=np.float64) / desc["fps"])
    save(prefix, "pixel_values_videos", f32(kw["pixel_values_videos"]))
    save(prefix, "video_position_ids", kw["video_position_ids"].cpu().numpy().astype(np.int64))
    soft = cap["soft_v"].reshape(-1, cap["soft_v"].shape[-1])
    per_frame = soft.shape[0] // n_frames
    assert per_frame * n_frames == soft.shape[0], (prefix, tuple(soft.shape), n_frames)
    save(prefix, "soft_per_frame", np.array([per_frame], dtype=np.int64))
    with open(os.path.join(snapshot_dir, "config.json")) as f:
        config = json.load(f)
    bos, eos = config["text_config"]["bos_token_id"], config["text_config"]["eos_token_id"]
    boi, eoi, vid = config["boi_token_id"], config["eoi_token_id"], config["video_token_id"]
    assert model[0].processor.video_token_id == vid, (model[0].processor.video_token_id, vid)
    if not desc.get("add_timestamps"):
        want_ids = [bos] + ([boi] + [vid] * per_frame + [eoi]) * n_frames + [eos]
        assert ids.tolist() == want_ids, (prefix, ids.tolist()[:20], want_ids[:20])
    return {"frames_sampled": n_frames, "soft_per_frame": per_frame, "indices": idx}


def main():
    path = snapshot()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    # (kind, n, tag, item, description); `tag` names a variant of the same input.
    inputs = [("image", n, "", make_image(w, h, s), {"width": w, "height": h, "seed": s})
              for n, (w, h, s) in enumerate(IMAGES)]
    inputs += [("audio", n, "", {"array": make_audio(sec, s), "sampling_rate": SR},
                {"seconds": sec, "seed": s, "sampling_rate": SR}) for n, (sec, s) in enumerate(AUDIOS)]
    for n, (t, h, w, fps, s) in enumerate(VIDEOS):
        # The metadata is what makes the processor sample at its fps: without a
        # frame rate and duration it keeps every frame (sentence-transformers
        # fills in fps None for a bare array, which is that path).
        item = {"array": make_video(t, h, w, s),
                "video_metadata": {"total_num_frames": t, "fps": fps, "duration": t / fps,
                                   "frames_indices": list(range(t))}}
        desc = {"frames": t, "height": h, "width": w, "fps": fps, "seed": s}
        inputs.append(("video", n, "", item, desc))
        if n == 0:
            inputs.append(("video", n, "t", item, {**desc, "add_timestamps": True}))

    m32 = SentenceTransformer(path, device="cpu",
                              model_kwargs={"attn_implementation": ATTN, "dtype": torch.float32})
    record = []
    with torch.no_grad():
        for kind, n, tag, item, desc in inputs:
            prefix = f"{kind}{n}{tag}_"
            set_timestamps(m32, desc.get("add_timestamps", False))
            emb, cap = run(m32, item)
            kw = cap["kwargs"]
            # Only the tower this input runs fires its layer hooks.
            expect = len(m32_eg(m32).audio_tower.layers if kind == "audio"
                         else m32_eg(m32).vision_tower.encoder.layers)
            assert len(cap["layers"]) == expect, (prefix, len(cap["layers"]), expect)
            ids = kw["input_ids"][0].cpu().numpy().astype(np.int64)
            save(prefix, "ids", ids)
            save(prefix, "attention_mask", kw["attention_mask"][0].cpu().numpy().astype(np.int64))
            # The raw input the processor received, for tessl's native preprocessing
            # (src/embedgemma2_pre.rs): the image as HWC RGB bytes (stored as int64,
            # the one integer dtype tessl's npy reader takes), the clip as mono
            # float32 at 16 kHz, the video as THWC bytes the same way.
            if kind == "image":
                save(prefix, "raw", np.asarray(item.convert("RGB"), dtype=np.uint8).astype(np.int64))
            elif kind == "video":
                save(prefix, "raw", item["array"].astype(np.int64))
            else:
                assert item["sampling_rate"] == SR
                save(prefix, "raw", np.asarray(item["array"], dtype=np.float32))
            extra = {}
            if kind == "video":
                extra = save_video(m32, prefix, kw, cap, desc, ids, path)
                soft = cap["soft_v"]
            elif kind == "image":
                save(prefix, "pixel_values", f32(kw["pixel_values"]))
                save(prefix, "image_position_ids", kw["image_position_ids"].cpu().numpy().astype(np.int64))
                save(prefix, "patch_embed", f32(cap["patch_embed"][0]))
                for i, t in enumerate(cap["layers"]):
                    save(prefix, f"layer{i}", f32(t[0]))
                save(prefix, "pooled", f32(cap["pooled"]))
                soft = cap["soft_v"]
            else:
                save(prefix, "input_features", f32(kw["input_features"]))
                save(prefix, "input_features_mask", kw["input_features_mask"].cpu().numpy().astype(np.int64))
                save(prefix, "subsample", f32(cap["subsample"][0]))
                for i, t in enumerate(cap["layers"]):
                    save(prefix, f"layer{i}", f32(t[0]))
                save(prefix, "tower_out", f32(cap["tower_out"][0]))
                mask = cap["out_mask"][0].bool()
                save(prefix, "out_mask", mask.cpu().numpy())
                soft = cap["soft_a"][0][mask]
            soft = soft.reshape(-1, soft.shape[-1])
            save(prefix, "soft", f32(soft))
            save(prefix, "lm_final", f32(cap["lm_final"][0]))
            save(prefix, "emb_f32", f32(emb))
            record.append({"kind": kind, "index": n, "prefix": prefix, **desc, **extra,
                           "tokens": int(kw["input_ids"].shape[1]), "soft_tokens": int(soft.shape[0]),
                           "layers_captured": len(cap["layers"])})
            print(prefix, record[-1])
        set_timestamps(m32, False)

    del m32
    m16 = SentenceTransformer(path, device="cpu", model_kwargs={"attn_implementation": ATTN})
    with torch.no_grad():
        for kind, n, tag, item, desc in inputs:
            set_timestamps(m16, desc.get("add_timestamps", False))
            emb, _ = run(m16, item)
            save(f"{kind}{n}{tag}_", "emb_bf16", f32(emb))
    with open(os.path.join(OUT, "inputs.json"), "w") as f:
        json.dump({"snapshot": path, "inputs": record}, f, indent=1)
        f.write("\n")
    with open(os.path.join(OUT, "provenance.json"), "w") as f:
        prov = provenance(path)
        prov["versions"]["pillow"] = importlib.metadata.version("pillow")
        prov["attn_implementation"] = ATTN
        prov["generator"] = "tools/embedgemma2_ref/make_mm_reference.py"
        json.dump(prov, f, indent=1)
        f.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
