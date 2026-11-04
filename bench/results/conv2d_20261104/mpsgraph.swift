// Throwaway baseline (nothing here ships): MPSGraph's convolution2D at SAM's 3x3 shapes, NCHW and
// NHWC, f16 and f32, as the strongest Apple baseline for tessl::conv. Times one graph execution of
// the conv alone (data already on the GPU), min and median over `rounds` runs after 3 warm-ups.
// Usage: mpsgraph <rounds>
import Foundation
import Metal
import MetalPerformanceShadersGraph

let rounds = Int(CommandLine.arguments[1])!
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let gdev = MPSGraphDevice(mtlDevice: dev)

struct Shape { let name: String; let c: Int; let o: Int; let h: Int; let w: Int }
let shapes = [
    Shape(name: "288^2 256->256", c: 256, o: 256, h: 288, w: 288),
    Shape(name: "144^2 256->256", c: 256, o: 256, h: 144, w: 144),
    Shape(name: "72^2 256->256", c: 256, o: 256, h: 72, w: 72),
    Shape(name: "1152^2 32->16", c: 32, o: 16, h: 1152, w: 1152),
]
print("shape\tlayout\tdtype\tmin_ms\tmedian_ms")
for s in shapes {
    for (layoutName, nchw) in [("NCHW", true), ("NHWC", false)] {
        for (dtName, dt) in [("f16", MPSDataType.float16), ("f32", MPSDataType.float32)] {
            let g = MPSGraph()
            let xShape: [NSNumber] = nchw ? [1, s.c, s.h, s.w].map { NSNumber(value: $0) } : [1, s.h, s.w, s.c].map { NSNumber(value: $0) }
            let wShape: [NSNumber] = [s.o, s.c, 3, 3].map { NSNumber(value: $0) }
            let x = g.placeholder(shape: xShape, dataType: dt, name: "x")
            let w = g.placeholder(shape: wShape, dataType: dt, name: "w")
            let d = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1,
                                                       groups: 1, paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1,
                                                       paddingStyle: .explicit, dataLayout: nchw ? .NCHW : .NHWC, weightsLayout: .OIHW)!
            let y = g.convolution2D(x, weights: w, descriptor: d, name: "y")
            let elem = dt == .float16 ? 2 : 4
            let xb = dev.makeBuffer(length: s.c * s.h * s.w * elem, options: .storageModePrivate)!
            let wb = dev.makeBuffer(length: s.o * s.c * 9 * elem, options: .storageModePrivate)!
            let xd = MPSGraphTensorData(xb, shape: xShape, dataType: dt)
            let wd = MPSGraphTensorData(wb, shape: wShape, dataType: dt)
            let exe = g.compile(with: gdev, feeds: [x: MPSGraphShapedType(shape: xShape, dataType: dt), w: MPSGraphShapedType(shape: wShape, dataType: dt)],
                                targetTensors: [y], targetOperations: nil, compilationDescriptor: nil)
            var times: [Double] = []
            for i in 0..<(rounds + 3) {
                let cb = MPSCommandBuffer(from: q)
                _ = exe.encode(to: cb, inputs: [xd, wd], results: nil, executionDescriptor: nil)
                cb.commit(); cb.waitUntilCompleted()
                if i >= 3 { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000) }
            }
            times.sort()
            print(String(format: "%@\t%@\t%@\t%.3f\t%.3f", s.name, layoutName, dtName, times[0], times[times.count / 2]))
        }
    }
}
