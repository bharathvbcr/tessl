// Throwaway probe (nothing here ships). Usage: probe <metallib> <rounds> <N> <CI> <CO> <H> <W> [kernel...]
// 3x3, stride 1, pad 1 conv. x NCHW f32 and w OIHW f32 are random; every kernel's output is checked
// against an f64 CPU reference at sampled output points (all points when the problem is small), then
// timed as GPU time of one command buffer holding `reps` back-to-back dispatches, min over rounds.
import Foundation
import Metal

let a = CommandLine.arguments
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(URL: URL(fileURLWithPath: a[1]))
let rounds = Int(a[2])!
let N = Int(a[3])!, CI = Int(a[4])!, CO = Int(a[5])!, H = Int(a[6])!, W = Int(a[7])!
let kernels = a.count > 8 ? Array(a[8...]) : ["mpp_conv_f16_o64_w32_h1"]
let P = 1, OH = H, OW = W
let XP = ProcessInfo.processInfo.environment["PROBE_PREPAD"] == "1" ? P : 0, Hp = H + 2 * XP, Wp = W + 2 * XP
func pso(_ n: String) -> MTLComputePipelineState {
    try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!)
}

func gpuMs(_ body: (MTLComputeCommandEncoder) -> Void) -> Double {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    body(e)
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    precondition(cb.status == .completed, "command buffer failed: \(String(describing: cb.error))")
    return (cb.gpuEndTime - cb.gpuStartTime) * 1000
}

var st: UInt64 = 0x2545F4914F6CDD1D
func rnd() -> Float { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return Float(st >> 40) / Float(1 << 23) - 1 }
// PROBE_DEBUG="kh,kw": x encodes its own index (c*10000 + h*100 + w, n = 0) and w is the identity at that
// one tap, so each output names the input element it read; the first mismatches are printed.
let debugTap = ProcessInfo.processInfo.environment["PROBE_DEBUG"].map { $0.split(separator: ",").map { Int($0)! } }
var xh = (0..<(N * CI * H * W)).map { _ in rnd() }
var wh = (0..<(CO * CI * 9)).map { _ in rnd() * 0.05 }
if let m = ProcessInfo.processInfo.environment["PROBE_MASK"] {
    // keep random weights only where the mask allows: "c<K" (input channels below K) or "tap=kh,kw"
    for o in 0..<CO { for c in 0..<CI { for kk in 0..<9 {
        let keep = m.hasPrefix("c<") ? c < Int(m.dropFirst(2))! : (kk == Int(m.split(separator: "=")[1].split(separator: ",")[0])! * 3 + Int(m.split(separator: ",")[1])!)
        if !keep { wh[(o * CI + c) * 9 + kk] = 0 }
    } } }
}
if let t = debugTap {
    for n in 0..<N { for c in 0..<CI { for h in 0..<H { for w in 0..<W { xh[((n * CI + c) * H + h) * W + w] = ProcessInfo.processInfo.environment["PROBE_SMALL"] == "1" ? Float((c % 64) * 256 + h * 16 + w) : Float(c * 10000 + h * 100 + w) } } } }
    wh = [Float](repeating: 0, count: CO * CI * 9)
    let perm = Int(ProcessInfo.processInfo.environment["PROBE_PERM"] ?? "1")!
    for o in 0..<CO { wh[((o * CI + (o * perm + (perm > 1 ? 3 : 0)) % CI) * 3 + t[0]) * 3 + t[1]] = 1 }
}
let xb = dev.makeBuffer(bytes: xh, length: xh.count * 4, options: .storageModeShared)!

// HWIO weights in f32 and f16
var whwio = [Float](repeating: 0, count: wh.count)
for o in 0..<CO { for c in 0..<CI { for kh in 0..<3 { for kw in 0..<3 {
    whwio[((kh * 3 + kw) * CI + c) * CO + o] = wh[((o * CI + c) * 3 + kh) * 3 + kw]
} } } }
let w32 = dev.makeBuffer(bytes: whwio, length: whwio.count * 4, options: .storageModeShared)!
let w16h = whwio.map { Float16($0) }
let w16 = dev.makeBuffer(bytes: w16h, length: w16h.count * 2, options: .storageModeShared)!
let xp32 = dev.makeBuffer(length: N * Hp * Wp * CI * 4, options: .storageModeShared)!
let xp16 = dev.makeBuffer(length: N * Hp * Wp * CI * 2, options: .storageModeShared)!
let y = dev.makeBuffer(length: N * OH * OW * CO * 4, options: .storageModeShared)!

func ref(_ n: Int, _ o: Int, _ oh: Int, _ ow: Int) -> Double {
    var s = 0.0
    for c in 0..<CI { for kh in 0..<3 { for kw in 0..<3 {
        let h = oh + kh - P, w = ow + kw - P
        if h < 0 || h >= H || w < 0 || w >= W { continue }
        s += Double(xh[((n * CI + c) * H + h) * W + w]) * Double(wh[((o * CI + c) * 3 + kh) * 3 + kw])
    } } }
    return s
}
// Sample points: all when small, else 4096 spread over the output plus every corner and edge row.
var pts: [(Int, Int, Int, Int)] = []
let total = N * CO * OH * OW
if total <= 200_000 {
    for n in 0..<N { for o in 0..<CO { for oh in 0..<OH { for ow in 0..<OW { pts.append((n, o, oh, ow)) } } } }
} else {
    for i in 0..<4096 { let v = (i &* 2654435761) % total; pts.append((v / (CO * OH * OW), v / (OH * OW) % CO, v / OW % OH, v % OW)) }
    for o in [0, CO - 1] { for (oh, ow) in [(0, 0), (0, OW - 1), (OH - 1, 0), (OH - 1, OW - 1)] { pts.append((N - 1, o, oh, ow)) } }
}
let refs = pts.map { ref($0.0, $0.1, $0.2, $0.3) }
var refSq = 0.0; for r in refs { refSq += r * r }

var dims = SIMD4<Int32>(Int32(CO), Int32(OW), Int32(OH), Int32(N))
var pdims = SIMD2<Int32>(Int32(Wp), Int32(Hp))
let adjEnv = (ProcessInfo.processInfo.environment["PROBE_ADJ"] ?? "0,0,0,0").split(separator: ",").map { Int32($0)! }
var adj = SIMD4<Int32>(adjEnv[0], adjEnv[1], adjEnv[2], adjEnv[3])
var pd = SIMD4<Int32>(Int32(CI), Int32(H), Int32(W), Int32(XP))
func encPad(_ e: MTLComputeCommandEncoder, half: Bool) {
    e.setComputePipelineState(pso(half ? "pad_nhwc_f16" : "pad_nhwc_f32"))
    e.setBuffer(xb, offset: 0, index: 0); e.setBuffer(half ? xp16 : xp32, offset: 0, index: 1)
    e.setBytes(&pd, length: 16, index: 2)
    e.dispatchThreads(MTLSize(width: N * Hp * Wp * CI, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}
func tile(_ k: String) -> (Int, Int, Int) {
    let p = k.split(separator: "_")
    let ot = Int(p.first { $0.hasPrefix("o") && Int($0.dropFirst()) != nil }!.dropFirst())!
    let wt = Int(p.first { $0.hasPrefix("w") && Int($0.dropFirst()) != nil }!.dropFirst())!
    let ht = Int(p.first { $0.hasPrefix("h") && Int($0.dropFirst()) != nil }!.dropFirst())!
    return (ot, wt, ht)
}
func encConv(_ e: MTLComputeCommandEncoder, _ k: String, _ p: MTLComputePipelineState) {
    let half = k.contains("f16")
    let (ot, wt, ht) = tile(k)
    e.setComputePipelineState(p)
    e.setBuffer(half ? xp16 : xp32, offset: 0, index: 0); e.setBuffer(half ? w16 : w32, offset: 0, index: 1)
    e.setBuffer(y, offset: 0, index: 2); e.setBytes(&dims, length: 16, index: 3); e.setBytes(&pdims, length: 8, index: 4); e.setBytes(&adj, length: 16, index: 5); var cin = Int32(CI); e.setBytes(&cin, length: 4, index: 6)
    e.dispatchThreadgroups(MTLSize(width: CO / ot, height: (OW + wt - 1) / wt, depth: N * ((OH + ht - 1) / ht)),
                           threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
}

let flops = 2.0 * Double(N * CO * OH * OW) * Double(CI * 9)
print("shape N\(N) CI\(CI) CO\(CO) \(H)x\(W) k3 p1; \(pts.count) checked points; \(String(format: "%.2f", flops / 1e9)) GFLOP")
for k in kernels {
    let p = pso(k)
    let half = k.contains("f16")
    memset(y.contents(), 0xff, y.length)   // NaN fill: an unwritten output fails the check
    _ = gpuMs { e in encPad(e, half: half); e.memoryBarrier(scope: .buffers); encConv(e, k, p) }
    let yp = y.contents().bindMemory(to: Float.self, capacity: N * OH * OW * CO)
    var errSq = 0.0, maxAbs = 0.0, bad = 0, shown = 0
    for (i, (n, o, oh, ow)) in pts.enumerated() {
        let g = Double(yp[((n * OH + oh) * OW + ow) * CO + o])
        if !g.isFinite { bad += 1; continue }
        let d = g - refs[i]; errSq += d * d; maxAbs = max(maxAbs, abs(d))
        if debugTap != nil && abs(d) > 0.5 && shown < 12 {
            shown += 1
            print("  miss n\(n) o\(o) h\(oh) w\(ow): got \(g) ref \(refs[i])")
        }
    }
    let reps = max(1, Int(2000.0 / max(1.0, flops / 1e9)))
    var best = Double.infinity, bestPad = Double.infinity
    for _ in 0..<rounds {
        best = min(best, gpuMs { e in for _ in 0..<reps { encConv(e, k, p) } } / Double(reps))
        bestPad = min(bestPad, gpuMs { e in for _ in 0..<reps { encPad(e, half: half) } } / Double(reps))
    }
    print(String(format: "%@  rel_l2 %.3e  max_abs %.3e  nonfinite %d  conv %.3f ms (%.1f TFLOP/s)  pad %.3f ms",
                 k, (errSq / refSq).squareRoot(), maxAbs, bad, best, flops / best / 1e9, bestPad))
}
