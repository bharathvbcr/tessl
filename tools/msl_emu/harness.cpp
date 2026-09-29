// Runs one tessl Qwen3.5 kernel on the CPU emulator.
//
//   harness <case_dir>
//
// `<case_dir>/params.txt` holds `kernel <name>`, `outputs <buf> ...`, and one
// `<key> <value>` line per scalar. Every other file `<buf>.bin` is a raw
// little-endian buffer, bound by name. Outputs are written back over their
// `.bin` files. The dispatch geometry below mirrors src/qwen35.rs exactly: a
// harness that launched a kernel differently from the host would be testing a
// different program.
#include <metal_stdlib>

#include "qwen35_attn.cpp"
#include "qwen35_gdn.cpp"
#include "qwen35_mlp.cpp"
#include "qwen35_score.cpp"
#include "flash_attn_rows.cpp"

#include <algorithm>
#include <csignal>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace metal;
using metal::emu::Ids;
using metal::emu::launch;

namespace {

std::string dir;
std::map<std::string, std::string> params;
std::map<std::string, std::vector<uint8_t>> bufs;

std::vector<uint8_t> &buf(const std::string &name) {
    auto it = bufs.find(name);
    if (it != bufs.end()) return it->second;
    std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "harness: missing buffer %s\n", name.c_str());
        std::exit(2);
    }
    // Sized exactly, never grown: a vector's spare capacity is memory
    // AddressSanitizer cannot see past, so an overrun into it would go unseen.
    f.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> data(size);
    f.read(reinterpret_cast<char *>(data.data()), std::streamsize(size));
    return bufs[name] = std::move(data);
}

float *F(const std::string &n) { return reinterpret_cast<float *>(buf(n).data()); }
uint *U(const std::string &n) { return reinterpret_cast<uint *>(buf(n).data()); }
bfloat *BF(const std::string &n) { return reinterpret_cast<bfloat *>(buf(n).data()); }

/// An optional buffer: absent means "bind something harmless", as the host
/// binds a placeholder when a flag says the kernel will not touch the slot.
float *Fopt(const std::string &n) {
    std::ifstream f(dir + "/" + n + ".bin");
    return f ? F(n) : F("__dummy");
}

uint P(const std::string &k) {
    auto it = params.find(k);
    if (it == params.end()) {
        std::fprintf(stderr, "harness: missing param %s\n", k.c_str());
        std::exit(2);
    }
    return uint(std::stoul(it->second));
}
float PF(const std::string &k) { return std::stof(params.at(k)); }

uint cdiv(uint a, uint b) { return (a + b - 1) / b; }

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: harness <case_dir>\n");
        return 2;
    }
    if (argc == 2 && std::string(argv[1]) == "--constants") {
        // The shapes the kernels are compiled for, for check_qwen35.py to hold
        // src/qwen35.rs's copies of them to.
        std::printf("GDN_DK %u\nGDN_C %u\nGDN_BV %u\nGDN_PREP_THREADS %u\nGDN_SCAN_THREADS %u\n"
                    "GDN_PREP_TG_FLOATS %u\nGDN_SCAN_TG_FLOATS %u\nGDN_SCAN16_TG_FLOATS %u\nGDN_REC_TG_FLOATS %u\n"
                    "REDUCE_MAX_SIMDGROUPS %u\nPREFIX_ATTN_D %u\nPREFIX_ATTN_R %u\nPREFIX_ATTN_SGT %u\n"
                    "PREFIX_DECODE_CHUNK %u\nPREFIX_DECODE_R %u\n",
                    GDN_DK, GDN_C, GDN_BV, GDN_PREP_THREADS, GDN_SCAN_THREADS, GDN_PREP_TG_FLOATS,
                    GDN_SCAN_TG_FLOATS, GDN_SCAN16_TG_FLOATS, GDN_REC_TG_FLOATS, REDUCE_MAX_SIMDGROUPS, PREFIX_ATTN_D, PREFIX_ATTN_R,
                    PREFIX_ATTN_SGT, PREFIX_DECODE_CHUNK, PREFIX_DECODE_R);
        return 0;
    }
    // A barrier deadlock is a failure, not a hang.
    alarm(600);
    dir = argv[1];
    std::ifstream pf(dir + "/params.txt");
    std::string line;
    std::vector<std::string> outputs;
    while (std::getline(pf, line)) {
        std::istringstream ss(line);
        std::string k;
        ss >> k;
        if (k == "outputs") {
            std::string o;
            while (ss >> o) outputs.push_back(o);
        } else if (!k.empty()) {
            std::string v;
            ss >> v;
            params[k] = v;
        }
    }
    bufs["__dummy"] = std::vector<uint8_t>(64, 0);
    const std::string kname = params.at("kernel");

    if (kname == "qwen35_conv1d_silu") {
        const uint B = P("B"), T = P("T"), C = P("C"), KW = P("KW"), ld_x = P("ld_x"), x_off = P("x_off"),
                   sb = P("state_bstride"), flags = P("flags");
        float *x = F("x"), *w = F("w"), *si = Fopt("state_in"), *y = F("y"), *so = Fopt("state_out");
        // Read only under flags & 4 (ragged rows), as the host binds it.
        uint *lens = reinterpret_cast<uint *>(Fopt("seq_lens"));
        launch(uint3(cdiv(C, 256), T + KW - 1, B), uint3(256, 1, 1), 0, [&](const Ids &id, float *) {
            qwen35_conv1d_silu(x, w, si, y, so, B, T, C, KW, ld_x, x_off, sb, flags, lens, id.gid);
        });
    } else if (kname == "qwen35_gdn_chunk") {
        // prep + scan, as `qwen35::gdn_chunk_forward` encodes them.
        const uint B = P("B"), T = P("T"), Hk = P("Hk"), Hv = P("Hv"), Dv = P("Dv"), ld_qkv = P("ld_qkv"),
                   q_off = P("q_off"), k_off = P("k_off"), v_off = P("v_off"), ld_ab = P("ld_ab"),
                   a_off = P("a_off"), b_off = P("b_off"), ld_out = P("ld_out"), out_off = P("out_off"),
                   sb = P("state_bstride"), flags = P("flags");
        const uint nc = cdiv(T, GDN_C);
        const size_t tp = size_t(nc) * GDN_C;
        std::vector<float> wk(size_t(B) * Hv * tp * GDN_DK), wq(wk.size()), wg(size_t(B) * Hv * tp),
            wb(wg.size()), ww(size_t(B) * Hv * nc * GDN_C * GDN_C), waq(ww.size());
        float *qkv = F("qkv"), *ab = F("ab"), *alog = F("a_log"), *dtb = F("dt_bias"), *out = F("out");
        float *si = Fopt("state_in"), *so = Fopt("state_out");
        uint *lens = reinterpret_cast<uint *>(Fopt("seq_lens"));
        const uint use_lens = (flags & 4u) != 0u ? 1u : 0u;
        launch(uint3(nc, Hv, B), uint3(GDN_PREP_THREADS, 1, 1), GDN_PREP_TG_FLOATS, [&](const Ids &id, float *tgm) {
            qwen35_gdn_chunk_prep(qkv, ab, alog, dtb, wk.data(), wq.data(), wg.data(), wb.data(), ww.data(),
                                  waq.data(), T, Hk, Hv, ld_qkv, q_off, k_off, ld_ab, a_off, b_off, lens, use_lens,
                                  tgm, id.tg, id.lid, id.sg, id.lane);
        });
        // `scan_bv16`: the 16-column-slice scan (qwen35_gdn_chunk_scan_bv16).
        const bool bv16 = params.count("scan_bv16") != 0;
        launch(uint3(Dv / (bv16 ? 16u : GDN_BV), Hv, B), uint3(GDN_SCAN_THREADS, 1, 1),
               bv16 ? GDN_SCAN16_TG_FLOATS : GDN_SCAN_TG_FLOATS, [&](const Ids &id, float *tgm) {
            if (bv16) {
                qwen35_gdn_chunk_scan_bv16(qkv, wk.data(), wq.data(), wg.data(), wb.data(), ww.data(), waq.data(),
                                           si, out, so, T, Hv, Dv, ld_qkv, v_off, ld_out, out_off, sb, flags, lens,
                                           tgm, id.tg, id.lid, id.sg, id.lane);
            } else {
                qwen35_gdn_chunk_scan(qkv, wk.data(), wq.data(), wg.data(), wb.data(), ww.data(), waq.data(), si,
                                      out, so, T, Hv, Dv, ld_qkv, v_off, ld_out, out_off, sb, flags, lens, tgm,
                                      id.tg, id.lid, id.sg, id.lane);
            }
        });
        if (params.count("dump_ws")) {
            bufs["ws_w"] = std::vector<uint8_t>((uint8_t *)ww.data(), (uint8_t *)(ww.data() + ww.size()));
            bufs["ws_aq"] = std::vector<uint8_t>((uint8_t *)waq.data(), (uint8_t *)(waq.data() + waq.size()));
            outputs.push_back("ws_w");
            outputs.push_back("ws_aq");
        }
    } else if (kname == "qwen35_gdn_recurrent") {
        const uint B = P("B"), T = P("T"), Hk = P("Hk"), Hv = P("Hv"), Dv = P("Dv"), ld_qkv = P("ld_qkv"),
                   q_off = P("q_off"), k_off = P("k_off"), v_off = P("v_off"), ld_ab = P("ld_ab"),
                   a_off = P("a_off"), b_off = P("b_off"), ld_out = P("ld_out"), out_off = P("out_off"),
                   sb = P("state_bstride"), flags = P("flags");
        float *qkv = F("qkv"), *ab = F("ab"), *alog = F("a_log"), *dtb = F("dt_bias"), *out = F("out");
        float *si = Fopt("state_in"), *so = params.count("in_place") ? si : Fopt("state_out");
        uint *lens = reinterpret_cast<uint *>(Fopt("seq_lens"));
        launch(uint3(Dv / GDN_BV, Hv, B), uint3(GDN_SCAN_THREADS, 1, 1), GDN_REC_TG_FLOATS, [&](const Ids &id, float *tgm) {
            qwen35_gdn_recurrent(qkv, ab, alog, dtb, si, out, so, T, Hk, Hv, Dv, ld_qkv, q_off, k_off, v_off,
                                 ld_ab, a_off, b_off, ld_out, out_off, sb, flags, lens, tgm, id.tg, id.sg,
                                 id.lane);
        });
    } else if (kname == "qwen35_gated_rms_norm_f32" || kname == "qwen35_gated_rms_norm_bf16") {
        const uint rows = P("rows"), H = P("H"), D = P("D"), ld_x = P("ld_x"), x_off = P("x_off"),
                   ld_z = P("ld_z"), z_off = P("z_off"), ld_out = P("ld_out"), out_off = P("out_off");
        const float eps = PF("eps");
        const uint per_tg = 8;
        const bool bf = kname.back() == '6';
        float *x = F("x"), *z = F("z"), *w = F("w");
        // Resolved before launch: `buf` inserts into a std::map, which the
        // kernel threads must never do concurrently.
        bfloat *out_bf = bf ? BF("out") : nullptr;
        float *out_f = bf ? nullptr : F("out");
        launch(uint3(cdiv(rows * H, per_tg), 1, 1), uint3(per_tg * 32, 1, 1), 0, [&](const Ids &id, float *) {
            if (bf) {
                qwen35_gated_rms_norm_bf16(x, z, w, out_bf, rows, H, D, ld_x, x_off, ld_z, z_off, ld_out,
                                           out_off, eps, id.tg.x, id.sg, id.lane, id.tptg);
            } else {
                qwen35_gated_rms_norm_f32(x, z, w, out_f, rows, H, D, ld_x, x_off, ld_z, z_off, ld_out,
                                          out_off, eps, id.tg.x, id.sg, id.lane, id.tptg);
            }
        });
    } else if (kname == "qwen35_attn_qk_norm_rope") {
        const uint B = P("B"), T = P("T"), Hq = P("Hq"), Hkv = P("Hkv"), D = P("D"), R = P("rotary_dim"),
                   ld_p = P("ld_p"), q_off = P("q_off"), k_off = P("k_off"), v_off = P("v_off"),
                   pos = P("pos_offset"), cap = P("kv_capacity"),
                   slot_base = params.count("slot_base") ? P("slot_base") : 0u,
                   pos_stride = params.count("pos_stride") ? P("pos_stride") : 0u;
        const float theta = PF("theta"), eps = PF("eps");
        const uint per_tg = 8;
        float *p = F("p"), *qw = F("q_norm_w"), *kw = F("k_norm_w"), *q = F("q_out"), *kc = F("k_cache"),
              *vc = F("v_cache");
        // `posbuf 1`: the device-buffer position variant, reading `pos_offset`
        // from a one-element u32 buffer instead of the scalar.
        const bool posbuf = params.count("posbuf") != 0;
        uint *pos_ptr = posbuf ? U("pos_buf") : nullptr;
        launch(uint3(cdiv(B * T * (Hq + 2 * Hkv), per_tg), 1, 1), uint3(per_tg * 32, 1, 1), 0,
               [&](const Ids &id, float *) {
                   if (posbuf) {
                       qwen35_attn_qk_norm_rope_posbuf(p, qw, kw, q, kc, vc, B, T, Hq, Hkv, D, R, ld_p, q_off, k_off,
                                                       v_off, pos_ptr, cap, theta, eps, slot_base, pos_stride, id.tg.x, id.sg,
                                                       id.lane,
                                                       id.tptg);
                   } else {
                       qwen35_attn_qk_norm_rope(p, qw, kw, q, kc, vc, B, T, Hq, Hkv, D, R, ld_p, q_off, k_off,
                                                v_off, pos, cap, theta, eps, slot_base, pos_stride, id.tg.x, id.sg,
                                                id.lane, id.tptg);
                   }
               });
    } else if (kname == "qwen35_attn_gate_f32" || kname == "qwen35_attn_gate_bf16") {
        const uint rows = P("rows"), Hq = P("Hq"), D = P("D"), ld_p = P("ld_p"), q_off = P("q_off"),
                   ld_out = P("ld_out"), out_off = P("out_off");
        const bool bf = kname.back() == '6';
        float *attn = F("attn"), *p = F("p");
        float *outf = bf ? nullptr : (params.count("in_place") ? attn : F("out"));
        bfloat *outb = bf ? BF("out") : nullptr;
        launch(uint3(cdiv(Hq * D, 32), rows, 1), uint3(32, 1, 1), 0, [&](const Ids &id, float *) {
            const uint2 gid(id.gid.x, id.gid.y);
            if (bf) {
                qwen35_attn_gate_bf16(attn, p, outb, rows, Hq, D, ld_p, q_off, ld_out, out_off, gid);
            } else {
                qwen35_attn_gate_f32(attn, p, outf, rows, Hq, D, ld_p, q_off, ld_out, out_off, gid);
            }
        });
    } else if (kname == "qwen35_swiglu_f32" || kname == "qwen35_swiglu_bf16") {
        const uint rows = P("rows"), width = P("width"), ld_gate = P("ld_gate"), gate_off = P("gate_off"),
                   ld_up = P("ld_up"), up_off = P("up_off"), ld_out = P("ld_out"), out_off = P("out_off");
        const bool bf = kname.back() == '6';
        float *gate = F("gate"), *up = F("up");
        float *outf = bf ? nullptr : F("out");
        bfloat *outb = bf ? BF("out") : nullptr;
        launch(uint3(cdiv(width, 32), rows, 1), uint3(32, 1, 1), 0, [&](const Ids &id, float *) {
            const uint2 gid(id.gid.x, id.gid.y);
            if (bf) {
                qwen35_swiglu_bf16(gate, up, outb, rows, width, ld_gate, gate_off, ld_up, up_off, ld_out, out_off,
                                   gid);
            } else {
                qwen35_swiglu_f32(gate, up, outf, rows, width, ld_gate, gate_off, ld_up, up_off, ld_out, out_off,
                                  gid);
            }
        });
    } else if (kname == "flash_attn_rows") {
        // The instantiation nn::flash_attn_rows picks for this head dim
        // (rows_lanes_for / rows_groups_for in src/nn.rs), and its grid.
        const uint B = P("B"), Tq = P("Tq"), H = P("H"), Hkv = P("Hkv"), D = P("D"), cap = P("kv_capacity");
        const float scale = PF("scale");
        uint R, G;
        decltype(&flash_attn_rows_h128_r8_g8) fn;
        if (D == 128) {
            R = 8, G = 8, fn = &flash_attn_rows_h128_r8_g8;
        } else if (D == 256) {
            R = 16, G = 32, fn = &flash_attn_rows_h256_r16_g32;
        } else {
            std::fprintf(stderr, "harness: flash_attn_rows has no D=%u default\n", D);
            return 2;
        }
        const uint rows_per_tg = G * (32 / R);
        float *q = F("q"), *k = F("k"), *v = F("v"), *o = F("o");
        uint *tkv = U("tkv"), *qpos = U("q_pos"), *kvpos = U("kv_pos");
        const uint window = 0, out_bf16 = 0;
        launch(uint3(cdiv(Tq, rows_per_tg), B * H, 1), uint3(G * 32, 1, 1), 0, [&](const Ids &id, float *) {
            fn(q, k, v, o, B, Tq, tkv, H, Hkv, window, scale, qpos, kvpos, out_bf16, cap,
               uint2(id.tg.x, id.tg.y), uint2(id.tid_in_tg.x, 0));
        });
    } else if (kname == "qwen35_attn_prefix_rows") {
        // qwen35::attn_prefix_rows' grid: x = ceil(Tq / rows per threadgroup),
        // y = B*H, SGT simdgroups.
        const uint B = P("B"), Tq = P("Tq"), H = P("H"), Hkv = P("Hkv"), prefix_len = P("P"),
                   suffix_cap = P("suffix_cap"),
                   row_stride = params.count("row_stride") ? P("row_stride") : 0u;
        const float scale = PF("scale");
        const uint rows_per_tg = PREFIX_ATTN_SGT * (32 / PREFIX_ATTN_R);
        float *q = F("q"), *kp = F("kp"), *vp = F("vp"), *ks = F("ks"), *vs = F("vs"), *o = F("o");
        uint *slen = U("suffix_len"), *qpos = U("q_pos");
        const uint out_bf16 = 0;
        launch(uint3(cdiv(Tq, rows_per_tg), B * H, 1), uint3(PREFIX_ATTN_SGT * 32, 1, 1), 0,
               [&](const Ids &id, float *) {
                   qwen35_attn_prefix_rows(q, kp, vp, ks, vs, o, Tq, prefix_len, slen, H, Hkv, scale, qpos,
                                           out_bf16, suffix_cap, row_stride, uint2(id.tg.x, id.tg.y),
                                           uint2(id.tid_in_tg.x, 0));
               });
    } else if (kname == "qwen35_attn_prefix_decode") {
        // qwen35::attn_prefix_decode: the partial pass over
        // ceil((P + suffix_cap) / CHUNK) x B*H/sgs threadgroups of sgs
        // simdgroups (the GQA group when it fits), then the reduce over B*H
        // threadgroups of 256 threads.
        const uint B = P("B"), H = P("H"), Hkv = P("Hkv"), prefix_len = P("P"), suffix_cap = P("suffix_cap"),
                   row_stride = params.count("row_stride") ? P("row_stride") : 0u;
        const float scale = PF("scale");
        const uint group = H / Hkv;
        const uint sgs = (group >= 1 && group <= 32 && H % group == 0) ? group : 1;
        const uint chunks = std::max(cdiv(prefix_len + suffix_cap, PREFIX_DECODE_CHUNK), 1u);
        float *q = F("q"), *kp = F("kp"), *vp = F("vp"), *ks = F("ks"), *vs = F("vs"), *o = F("o");
        uint *slen = U("suffix_len"), *qpos = U("q_pos");
        std::vector<float> scratch(size_t(B) * H * chunks * (PREFIX_ATTN_D + 2));
        float *part = scratch.data();
        const uint out_bf16 = 0;
        launch(uint3(chunks, B * (H / sgs), 1), uint3(sgs * 32, 1, 1), 0, [&](const Ids &id, float *) {
            qwen35_attn_prefix_decode_partial(q, kp, vp, ks, vs, part, prefix_len, slen, H, Hkv, scale, qpos,
                                              suffix_cap, row_stride, uint2(id.tg.x, id.tg.y),
                                              uint2(id.tid_in_tg.x, 0), uint2(sgs * 32, 1));
        });
        launch(uint3(1, B * H, 1), uint3(256, 1, 1), 0, [&](const Ids &id, float *) {
            qwen35_attn_prefix_decode_reduce(part, o, prefix_len, slen, H, out_bf16, suffix_cap, row_stride,
                                             uint2(id.tg.x, id.tg.y), uint2(id.tid_in_tg.x, 0), uint2(256, 1));
        });
    } else if (kname == "qwen35_embed_rows_bf16") {
        // qwen35::embed_rows: dispatch_2d over (hidden, n), 32-wide rows.
        const uint n = P("n"), hidden = P("hidden"), vocab = P("vocab");
        uint *ids = U("ids");
        ushort *table = reinterpret_cast<ushort *>(buf("table").data());
        float *out = F("out");
        launch(uint3(cdiv(hidden, 32), n, 1), uint3(32, 1, 1), 0, [&](const Ids &id, float *) {
            qwen35_embed_rows_bf16(ids, table, out, n, hidden, vocab, uint2(id.gid.x, id.gid.y));
        });
    } else if (kname == "qwen35_score_rows_f32" || kname == "qwen35_score_rows_bf16") {
        const uint rows = P("rows"), hidden = P("hidden"), n_ans = P("n_ans"), vocab = P("vocab"),
                   n_slots = P("n_slots");
        const float eps = PF("eps"), w_offset = PF("w_offset");
        const bool bf = kname.back() == '6';
        float *h = F("h"), *nw = F("norm_w"), *lg = F("logits"), *lp = F("logprobs");
        uint *slots = U("slots"), *ans = U("answers");
        bfloat *emb_bf = bf ? BF("emb") : nullptr;
        float *emb_f = bf ? nullptr : F("emb");
        launch(uint3(n_slots, 1, 1), uint3(256, 1, 1), REDUCE_MAX_SIMDGROUPS + n_ans,
               [&](const Ids &id, float *tgm) {
                   if (bf) {
                       qwen35_score_rows_bf16(h, slots, nw, emb_bf, ans, lg, lp, rows, hidden, n_ans, vocab, eps,
                                              w_offset, tgm, id.tg.x, id.lid, id.sg, id.lane, id.tptg);
                   } else {
                       qwen35_score_rows_f32(h, slots, nw, emb_f, ans, lg, lp, rows, hidden, n_ans, vocab, eps,
                                             w_offset, tgm, id.tg.x, id.lid, id.sg, id.lane, id.tptg);
                   }
               });
    } else {
        std::fprintf(stderr, "harness: unknown kernel %s\n", kname.c_str());
        return 2;
    }

    for (const auto &o : outputs) {
        auto &b = buf(o);
        std::ofstream f(dir + "/" + o + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char *>(b.data()), std::streamsize(b.size()));
    }
    return 0;
}
