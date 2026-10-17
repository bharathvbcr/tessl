//! The device memory of `EmbedGemma2Model::encode` on the tiny checkpoint
//! (`tests/common/embedgemma2.rs`), in both `GemmOperands` modes: how many
//! bytes one call's activations take, and that consecutive calls reuse them.
//! The real-checkpoint figures are `bench/results/embedgemma2_memory_m5pro.txt`.
//!
//! Bounds: a second, third and fourth `encode` of the same batch make no fresh
//! device allocation at all (`extra == 0`): since the pool recycles a dropped
//! buffer at the next allocation when the GPU is idle (`recycle_if_idle`), the
//! previous call's output buffer (`ForwardIo::out`, dropped after its wait) is
//! reused too. It was one output above the first call's peak before that, a
//! bound these tests used to allow (`extra <= out_bucket`), and with the
//! recycle stubbed out every consecutive-call test here fails by exactly one
//! output buffer. The outputs are made larger than 16 KiB, from which the
//! device charges a buffer's own length, so that one output is visible in
//! `current_allocated_bytes`. A call that starts beside open work, where the
//! pool cannot recycle, still reuses everything the hand-back returned before
//! the previous wait (`an_encode_beside_open_work_reuses_the_previous_calls_set`).
//! The bytes one call allocates equal the itemized set the
//! `Acts` documentation lists (exact, from [`GpuRuntime::allocated_bytes_for`])
//! plus the call's own inputs and output, with `q|k|v` and `gate|up` sharing a
//! buffer.

mod common;

use common::embedgemma2::{open, safetensors_bytes, weights_for, CONFIG, PREFIX};
use common::embedgemma2_mm::{checkpoint_bytes, clip, config_json, configs, Configs, Image, AUDIO, IMAGE};
use common::{in_child, with_gpu};
use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model, PLE_RUN_BYTES};
use tessl::embedgemma2_audio::{AudioInput, EmbedGemma2Audio};
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput};
use tessl::gemm::GemmOperands;
use tessl::{BufferKind, GpuRuntime};

const MODES: [GemmOperands; 2] = [GemmOperands::ExactF32, GemmOperands::Bf16];

/// Three sequences of 40 tokens: one forward of 120 rows.
fn batch(vocab: u32) -> Vec<Vec<u32>> {
    batch_of(vocab, 3)
}

/// `n` sequences of 40 tokens.
fn batch_of(vocab: u32, n: u32) -> Vec<Vec<u32>> {
    (0..n)
        .map(|s| (0..40u32).map(|i| (i * 5 + s * 3 + 1) % vocab).collect())
        .collect()
}

fn cold(n: usize) -> u64 {
    GpuRuntime::allocated_bytes_for(n, BufferKind::Cold)
}

/// A second, third and fourth `encode` of the batch hold one set of
/// activations, not two. `encode` dropped its activations after its last wait,
/// and the pool recycles a dropped buffer at the next commit, which the next
/// `encode` reaches after it has allocated its own: each call then allocated a
/// whole second set (two live across consecutive calls, 1472 MiB at 8192
/// rows). `encode` now releases them before its wait, so the wait recycles them.
#[test]
fn consecutive_encodes_hold_one_activation_set() {
    in_child(
        "consecutive_encodes_hold_one_activation_set",
        &[],
        consecutive_encodes_hold_one_activation_set_in_process,
    );
}

fn consecutive_encodes_hold_one_activation_set_in_process() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    // 40 sequences: the output (40 x 128 x 4 bytes) is past the 16 KiB from
    // which the device charges a buffer's own length, so a second output that
    // was not reused would move the figure this test reads.
    let seqs = batch_of(cfg.vocab, 40);
    let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
    let out_bucket = cold(refs.len() * cfg.embedding_dim as usize * 4);
    assert!(
        out_bucket >= 16 << 10,
        "test bug: the output is under the 16 KiB charge threshold"
    );
    for operands in MODES {
        let w = weights_for(&cfg, 41, operands);
        let st = open("memory_consecutive", safetensors_bytes(&w, &[]));
        with_gpu(|rt| {
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
            rt.synchronize().unwrap();
            let base = rt.current_allocated_bytes();
            let first = model.encode(&refs, None, false).unwrap();
            assert_eq!(first.forwards, 1, "test bug: the batch split");
            let after_first = rt.current_allocated_bytes();
            let set = after_first - base;
            assert!(
                set > 100 * out_bucket,
                "{operands:?}: the first call allocated {set} bytes: too few to tell a set from its output"
            );
            rt.reset_peak_allocated_bytes();
            for call in 2..=4 {
                let again = model.encode(&refs, None, false).unwrap();
                assert_eq!(
                    again.embeddings.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    first.embeddings.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{operands:?}: call {call} changed a bit"
                );
            }
            // None: the pool recycles the previous call's dropped output at
            // the next call's first allocation (`recycle_if_idle`), so a
            // repeated call reuses the whole set, its output included. Before
            // that, the previous output waited for a commit and a repeated call
            // reached one more `out_bucket` bytes.
            let extra = rt.peak_allocated_bytes() - after_first;
            assert_eq!(
                extra, 0,
                "{operands:?}: three more calls reached {extra} bytes above the first ({set} bytes of activations, \
                 one {out_bucket}-byte output); a repeated call must reuse the first call's buffers exactly"
            );
        });
    }
}

/// The pool recycles a dropped buffer at the next allocation only when no
/// command buffer is open and the GPU has caught up (`recycle_if_idle`). A call
/// that starts while its caller has work open in an `async_scope` gets neither,
/// so a set that was only *dropped* after the previous call's wait would not be
/// reused, and the call would allocate a second set. `encode` hands its set back
/// before its wait instead, so the wait returns it to the freelist and the next
/// call reuses it whatever the caller has open: it reaches at most one fresh
/// output above the first call's peak, not a second set. (This is why the
/// hand-back stays beside the pool's recycle-at-allocation: with the pool fix
/// alone, back-to-back calls are the same, and this case is not.)
#[test]
fn an_encode_beside_open_work_reuses_the_previous_calls_set() {
    in_child(
        "an_encode_beside_open_work_reuses_the_previous_calls_set",
        &[],
        an_encode_beside_open_work_reuses_the_previous_calls_set_in_process,
    );
}

fn an_encode_beside_open_work_reuses_the_previous_calls_set_in_process() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let seqs = batch_of(cfg.vocab, 40);
    let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
    let out_bucket = cold(refs.len() * cfg.embedding_dim as usize * 4);
    for operands in MODES {
        let w = weights_for(&cfg, 41, operands);
        let st = open("memory_beside_open_work", safetensors_bytes(&w, &[]));
        with_gpu(|rt| {
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
            let (src, dst) = (
                rt.alloc_tensor_f32(&[1024]).unwrap(),
                rt.alloc_tensor_f32(&[1024]).unwrap(),
            );
            rt.synchronize().unwrap();
            let base = rt.current_allocated_bytes();
            let first = model.encode(&refs, None, false).unwrap();
            let after_first = rt.current_allocated_bytes();
            let set = after_first - base;
            assert!(
                set > 100 * out_bucket,
                "{operands:?}: the first call allocated {set} bytes"
            );
            rt.reset_peak_allocated_bytes();
            // Open work: a copy encoded into an open scope's command buffer, not
            // yet waited for, when the second call allocates.
            let scope = rt.async_scope();
            tessl::tensor::gpu_copy(&src, &dst).unwrap();
            let again = model.encode(&refs, None, false).unwrap();
            drop(scope);
            rt.synchronize().unwrap();
            assert_eq!(
                bits_of(&again.embeddings),
                bits_of(&first.embeddings),
                "{operands:?}: the second call changed a bit"
            );
            let extra = rt.peak_allocated_bytes() - after_first;
            // Exactly one fresh output: the first call's output is dropped
            // after its wait, and with a command buffer open the allocation
            // cannot recycle it. Everything handed back before the wait is
            // reused; without the hand-back this is the whole set again.
            assert_eq!(
                extra, out_bucket,
                "{operands:?}: a call beside open work reached {extra} bytes above the first ({set} bytes of \
                 activations, one {out_bucket}-byte output): exactly the output is fresh; more is a second set"
            );
        });
    }
}

/// FNV-1a over the bit patterns of `v`.
fn fnv1a_bits(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        x.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
    })
}

/// Prints the FNV-1a hash of the tiny checkpoint's embeddings, per mode, for a
/// one-forward batch, a two-forward batch (a 4000-token sequence beside seven
/// of 30) and the first with a Matryoshka prefix, each from three consecutive
/// calls (which must agree). The tool for "a memory change moved no output
/// bit": run it on two builds and compare the `TINY` lines.
///
/// `cargo test --release --test embedgemma2_memory -- --ignored --nocapture tiny_embedding_hashes`
#[test]
#[ignore = "prints hashes for comparing two builds; asserts only that repeated calls agree"]
fn tiny_embedding_hashes() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let three = batch(cfg.vocab);
    let long: Vec<u32> = (0..4000u32).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let short: Vec<Vec<u32>> = (0..7u32)
        .map(|s| (0..30u32).map(|i| (i * 5 + s) % cfg.vocab).collect())
        .collect();
    let mut ragged: Vec<&[u32]> = vec![&long];
    ragged.extend(short.iter().map(Vec::as_slice));
    let three: Vec<&[u32]> = three.iter().map(Vec::as_slice).collect();
    for operands in MODES {
        let w = weights_for(&cfg, 47, operands);
        let st = open("memory_hashes", safetensors_bytes(&w, &[]));
        with_gpu(|rt| {
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
            for (name, b, dim) in [
                ("three", &three, None),
                ("ragged", &ragged, None),
                ("three-trunc48", &three, Some(48)),
            ] {
                let hashes: Vec<u64> = (0..3)
                    .map(|_| fnv1a_bits(&model.encode(b, dim, false).unwrap().embeddings))
                    .collect();
                assert!(
                    hashes.iter().all(|&h| h == hashes[0]),
                    "{operands:?} {name}: {hashes:x?}"
                );
                println!("TINY {operands:?} {name} {:016x}", hashes[0]);
            }
        });
    }
}

/// One `encode`'s device bytes are exactly the itemized activation set plus
/// the call's inputs and output: the buffers listed on `Acts`, with the
/// `q|k|v` GEMM's output and the MLP's `gate|up` in one buffer sized for the
/// wider, the per-layer-input run at the library's bound, and under `Bf16`
/// the bf16 operand copies and half-width Q, K, V and per-layer gate. A
/// buffer added to the set, or one that stops sharing, moves the total.
#[test]
fn one_encode_allocates_the_itemized_set() {
    in_child(
        "one_encode_allocates_the_itemized_set",
        &[],
        one_encode_allocates_the_itemized_set_in_process,
    );
}

fn one_encode_allocates_the_itemized_set_in_process() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let seqs = batch(cfg.vocab);
    let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
    let (nb, seq) = (refs.len(), refs[0].len());
    let r = nb * seq;
    let (h, ple, inter, dim) = (
        cfg.hidden as usize,
        cfg.ple_dim as usize,
        cfg.intermediate as usize,
        cfg.embedding_dim as usize,
    );
    let (mut qkv_w, mut q_w, mut kv_w) = (0usize, 0usize, 0usize);
    for s in &cfg.layers {
        let (q, kv, d) = (cfg.q_heads as usize, s.kv_heads as usize, s.head_dim as usize);
        qkv_w = qkv_w.max((q + 2 * kv) * d);
        q_w = q_w.max(q * d);
        kv_w = kv_w.max(kv * d);
    }
    let ple_run = (PLE_RUN_BYTES / 4).max(r * ple).min(r * ple * cfg.layers.len());
    for operands in MODES {
        let bf16 = operands == GemmOperands::Bf16;
        let (f, b) = (4usize, 2usize);
        let qkv_dtype = if bf16 { b } else { f };
        let mut items: Vec<usize> = vec![
            r * h * f,                                       // embeds
            r * h * f,                                       // resid
            r * h * f,                                       // x
            r * qkv_w.max(2 * inter) * f,                    // q|k|v and gate|up, one buffer
            r * q_w * qkv_dtype,                             // q
            r * kv_w * qkv_dtype,                            // k
            r * kv_w * qkv_dtype,                            // v
            r * if bf16 { q_w } else { q_w.max(inter) } * f, // attn (+ the f32 MLP mid)
            r * h * f,                                       // y
            ple_run * f,                                     // per-layer inputs
            r * ple * f,                                     // ple_gate
            r * ple * qkv_dtype,                             // ple_mid
            nb * h * f,                                      // pooled
        ];
        if bf16 {
            items.extend([r * h * b, r * q_w.max(inter) * b, r * h * b, nb * h * b]);
        }
        // The call's own: ids, lens, segments, embeddings.
        items.extend([r * 4, nb * 4, nb * 2 * 4, nb * dim * 4]);
        // `allocated_bytes_for` is what the device charges from 16 KiB up; a
        // smaller buffer may be charged less than its power-of-two length, so
        // those bound the total from above only.
        let (big, small): (Vec<u64>, Vec<u64>) = items.iter().map(|&n| cold(n)).partition(|&n| n >= 16 << 10);
        let (want, slack) = (big.iter().sum::<u64>(), small.iter().sum::<u64>());
        let w = weights_for(&cfg, 43, operands);
        let st = open("memory_itemized", safetensors_bytes(&w, &[]));
        // Pipelines are compiled on a kernel's first use and charged to the
        // device outside the pool (256 KiB here), once per process. Whether an
        // earlier test in this binary paid that depended on test order, so one
        // warm-up encode on a runtime of its own pays it first; the bytes
        // asserted are the second runtime's.
        with_gpu(|rt| {
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
            model.encode(&refs, None, false).unwrap();
        });
        with_gpu(|rt| {
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
            rt.synchronize().unwrap();
            let base = rt.current_allocated_bytes();
            model.encode(&refs, None, false).unwrap();
            let got = rt.current_allocated_bytes() - base;
            assert!(
                (want..=want + slack).contains(&got),
                "{operands:?}: one encode took {got} bytes, the itemized set is {want} (+ up to {slack} \
                 for buffers under 16 KiB)"
            );
        });
    }
}

// ---------------------------------------------------------------------------
// The towers and the facade (tiny checkpoint of `tests/common/embedgemma2_mm.rs`)
// ---------------------------------------------------------------------------
//
// Bounds: consecutive tower calls of one forward each hold one activation set,
// and calls 2 to 4 reach exactly the first call's peak (`extra == 0`): the
// previous call's `soft` output is recycled at the next allocation, as is the
// facade's text-encoder output. (Before the pool recycled at allocation, each
// reached one output above; the bound was `extra <= out_bucket`.)

fn tower_media(c: &Configs) -> (Vec<Image>, Vec<AudioInput>) {
    let pd = c.vision.patch_dim() as usize;
    (
        vec![
            Image::new(8, 6, 16, pd, 1),
            Image::new(5, 4, 28, pd, 2),
            Image::new(1, 1, 3, pd, 3),
            Image::new(2, 2, 0, pd, 4),
        ],
        vec![
            clip(&c.audio, 41, 1, |_| true),
            clip(&c.audio, 58, 2, |f| f < 45),
            clip(&c.audio, 30, 3, |f| !(8..16).contains(&f)),
        ],
    )
}

fn bits_of(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Four vision calls in a row hold one activation set, in both modes. The
/// tower dropped its activations and inputs at `collect`, after its wait, so
/// the pool recycled them at the next call's first commit, after that call had
/// allocated its own: two sets were live across consecutive calls. It now
/// releases them before the wait, so the wait recycles them.
#[test]
fn consecutive_vision_calls_hold_one_activation_set() {
    in_child(
        "consecutive_vision_calls_hold_one_activation_set",
        &[],
        consecutive_vision_calls_hold_one_activation_set_in_process,
    );
}

fn consecutive_vision_calls_hold_one_activation_set_in_process() {
    let c = configs();
    let (images, _) = tower_media(&c);
    // The four images four times over: the soft output is past the 16 KiB from
    // which the device charges a buffer's own length (see the text test).
    let inputs: Vec<ImageInput<'_>> = (0..4).flat_map(|_| images.iter().map(Image::input)).collect();
    for operands in MODES {
        let st = open("memory_vision", checkpoint_bytes(&c, 17, operands));
        with_gpu(|rt| {
            let model = EmbedGemma2VisionModel::load_with(rt, &st, c.vision.clone(), operands).unwrap();
            rt.synchronize().unwrap();
            let soft_rows: usize = model.check_images(&inputs).unwrap().iter().sum();
            let out_bucket = cold(soft_rows * c.vision.text_hidden as usize * 4);
            assert!(
                out_bucket >= 16 << 10,
                "test bug: the output is under the 16 KiB charge threshold"
            );
            let base = rt.current_allocated_bytes();
            let first = model.encode_images(&inputs, false).unwrap();
            assert_eq!(first.forwards, 1, "test bug: the call split");
            let after_first = rt.current_allocated_bytes();
            let set = after_first - base;
            assert!(
                set > 20 * out_bucket,
                "{operands:?}: the first call allocated {set} bytes: too few to tell a set from its output"
            );
            rt.reset_peak_allocated_bytes();
            for call in 2..=4 {
                let again = model.encode_images(&inputs, false).unwrap();
                for (a, b) in again.soft.iter().zip(&first.soft) {
                    assert_eq!(bits_of(a), bits_of(b), "{operands:?}: call {call} changed a bit");
                }
            }
            let extra = rt.peak_allocated_bytes() - after_first;
            assert_eq!(
                extra, 0,
                "{operands:?}: three more vision calls reached {extra} bytes above the first ({set} bytes of \
                 activations, one {out_bucket}-byte output); a repeated call must reuse the first call's buffers exactly"
            );
        });
    }
}

/// The same for the audio tower, in both modes (`Bf16` also holds its operands
/// in bf16 buffers of the same set).
#[test]
fn consecutive_audio_calls_hold_one_activation_set() {
    in_child(
        "consecutive_audio_calls_hold_one_activation_set",
        &[],
        consecutive_audio_calls_hold_one_activation_set_in_process,
    );
}

fn consecutive_audio_calls_hold_one_activation_set_in_process() {
    let c = configs();
    let (_, clips) = tower_media(&c);
    for operands in MODES {
        let st = open("memory_audio", checkpoint_bytes(&c, 17, operands));
        with_gpu(|rt| {
            let model = EmbedGemma2Audio::load_with(rt, &st, "", c.audio.clone(), operands).unwrap();
            rt.synchronize().unwrap();
            let base = rt.current_allocated_bytes();
            let first = model.encode_audio(&clips, false).unwrap();
            assert_eq!(first.forwards, 1, "test bug: the call split");
            // The forward's output holds every tower row, kept or not.
            let rows: usize = first.out_masks.iter().map(Vec::len).sum();
            let out_bucket = cold(rows * c.audio.embed_dim as usize * 4);
            let after_first = rt.current_allocated_bytes();
            let set = after_first - base;
            assert!(
                set > 20 * out_bucket,
                "{operands:?}: the first call allocated {set} bytes: too few to tell a set from its output"
            );
            rt.reset_peak_allocated_bytes();
            for call in 2..=4 {
                let again = model.encode_audio(&clips, false).unwrap();
                for (a, b) in again.soft_tokens.iter().zip(&first.soft_tokens) {
                    assert_eq!(bits_of(a), bits_of(b), "{operands:?}: call {call} changed a bit");
                }
            }
            let extra = rt.peak_allocated_bytes() - after_first;
            assert_eq!(
                extra, 0,
                "{operands:?}: three more audio calls reached {extra} bytes above the first ({set} bytes of \
                 activations, one {out_bucket}-byte output); a repeated call must reuse the first call's buffers exactly"
            );
        });
    }
}

/// Consecutive mixed calls of the facade (text, an image, a clip and a
/// caller-built item holding two images and a clip) hold one set of each
/// model's activations: calls 2 to 4 reach at most one fresh text-encoder
/// output above the first. The towers' soft rows stay on the device between
/// the towers and the text encoder and are released with everything else
/// before the call's one wait.
#[test]
fn consecutive_facade_calls_hold_one_activation_set() {
    in_child(
        "consecutive_facade_calls_hold_one_activation_set",
        &[],
        consecutive_facade_calls_hold_one_activation_set_in_process,
    );
}

fn consecutive_facade_calls_hold_one_activation_set_in_process() {
    let c = configs();
    let (images, clips) = tower_media(&c);
    let text_ids: Vec<u32> = (0..24u32).map(|i| 3 + i % 40).collect();
    for operands in MODES {
        let st = open("memory_facade", checkpoint_bytes(&c, 17, operands));
        with_gpu(|rt| {
            let m = EmbedGemma2Multimodal::load_from(rt, &st, &config_json(), operands).unwrap();
            let imgs = [images[0].input(), images[1].input()];
            let one_clip = std::slice::from_ref(&clips[0]);
            let tokens_i = m.vision().check_images(&imgs).unwrap();
            let tokens_a = m.audio().check_clips(one_clip).unwrap()[0];
            let mut ids = vec![1u32];
            ids.extend(std::iter::repeat_n(IMAGE, tokens_i[0] + tokens_i[1]));
            ids.extend(std::iter::repeat_n(AUDIO, tokens_a));
            ids.push(2);
            let mut items = vec![
                MmItem::Text(&text_ids),
                MmItem::Image(images[2].input()),
                MmItem::Audio(&clips[1]),
                MmItem::Media {
                    ids: &ids,
                    images: &imgs,
                    audio: one_clip,
                    videos: &[],
                },
            ];
            // Forty more texts: the output (44 x 128 x 4 bytes) is past the
            // 16 KiB from which the device charges a buffer's own length.
            items.extend((0..40).map(|_| MmItem::Text(&text_ids)));
            let out_bucket = cold(items.len() * m.text().config().embedding_dim as usize * 4);
            assert!(
                out_bucket >= 16 << 10,
                "test bug: the output is under the 16 KiB charge threshold"
            );
            rt.synchronize().unwrap();
            let base = rt.current_allocated_bytes();
            let first = m.encode(&items, None).unwrap();
            let after_first = rt.current_allocated_bytes();
            let set = after_first - base;
            assert!(
                set > 20 * out_bucket,
                "{operands:?}: the first call allocated {set} bytes: too few to tell a set from its output"
            );
            rt.reset_peak_allocated_bytes();
            for call in 2..=4 {
                let again = m.encode(&items, None).unwrap();
                assert_eq!(
                    bits_of(&again.embeddings),
                    bits_of(&first.embeddings),
                    "{operands:?}: call {call} changed a bit"
                );
            }
            let extra = rt.peak_allocated_bytes() - after_first;
            assert_eq!(
                extra, 0,
                "{operands:?}: three more facade calls reached {extra} bytes above the first ({set} bytes of \
                 activations, one {out_bucket}-byte output); a repeated call must reuse the first call's buffers exactly"
            );
        });
    }
}
