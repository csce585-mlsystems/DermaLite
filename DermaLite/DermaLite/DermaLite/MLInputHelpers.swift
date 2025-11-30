import UIKit
import CoreML
import Accelerate

// MARK: - Convert UIImage -> CVPixelBuffer (useful for Image inputs)
extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        self.draw(in: CGRect(origin: .zero, size: size))
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return out
    }

    func toCVPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        guard let cgImage = self.cgImage else { return nil }
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard status == kCVReturnSuccess, let px = pixelBuffer else { return nil }
        CVPixelBufferLockBaseAddress(px, [])
        defer { CVPixelBufferUnlockBaseAddress(px, []) }

        let pxData = CVPixelBufferGetBaseAddress(px)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = CVPixelBufferGetBytesPerRow(px)
        guard let ctx = CGContext(data: pxData, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: rgbColorSpace,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            return nil
        }
        // flip context vertically
        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1.0, y: -1.0)
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return px
    }
}


// MARK: - Convert UIImage -> MLMultiArray (Float32) with normalization
/// Produces an MLMultiArray with shape [1, 3, H, W] (batch-first). Adjust if your model expects [3,H,W] or other order.
func imageToMLMultiArray(_ image: UIImage,
                         width: Int = 224,
                         height: Int = 224,
                         mean: (Float,Float,Float) = (0.485, 0.456, 0.406),
                         std: (Float,Float,Float)  = (0.229, 0.224, 0.225),
                         batchFirst: Bool = true) -> MLMultiArray? {

    guard let resized = image.resized(to: CGSize(width: width, height: height)),
          let cgImage = resized.cgImage else { return nil }

    // Create buffer for pixel data (ARGB)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesPerRow = 4 * width
    let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
    guard let ctx = CGContext(data: nil, width: width, height: height,
                              bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                              space: colorSpace, bitmapInfo: bitmapInfo),
          let data = ctx.data else { return nil }

    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    // Allocate MLMultiArray: shape [1,3,H,W] or [3,H,W]
    let shape: [NSNumber] = batchFirst ? [1, 3, NSNumber(value: height), NSNumber(value: width)] : [3, NSNumber(value: height), NSNumber(value: width)]
    guard let mlArray = try? MLMultiArray(shape: shape, dataType: .float32) else { return nil }

    // Fill array: CoreML uses row-major indexing for MLMultiArray; we will set value using linear index
    let ptr = data.bindMemory(to: UInt8.self, capacity: bytesPerRow * height)

    // Iterate over pixels and write normalized floats
    // Pixel layout in ctx is ARGB (alpha, red, green, blue)
    let meanR = mean.0, meanG = mean.1, meanB = mean.2
    let stdR = std.0, stdG = std.1, stdB = std.2

    // Access MLMultiArray's pointer
    if mlArray.dataType != .float32 { return nil }
    let count = mlArray.count
    let floatPtr = mlArray.dataPointer.bindMemory(to: Float.self, capacity: count)

    // We'll write as [B, C, H, W] with channel-major block to match many conversions.
    // Layout: index = ((b * C + c) * H + y) * W + x
    let C = 3
    let H = height
    let W = width

    for y in 0..<H {
        for x in 0..<W {
            let pixelIndex = y * bytesPerRow + x * 4
            let a = ptr[pixelIndex]      // alpha (unused)
            let r = ptr[pixelIndex + 1]
            let g = ptr[pixelIndex + 2]
            let b = ptr[pixelIndex + 3]

            // normalize to [0,1]
            let rf = (Float(r) / 255.0 - meanR) / stdR
            let gf = (Float(g) / 255.0 - meanG) / stdG
            let bf = (Float(b) / 255.0 - meanB) / stdB

            if batchFirst {
                // channel 0 = R
                let idxR = ((0 * C + 0) * H + y) * W + x
                let idxG = ((0 * C + 1) * H + y) * W + x
                let idxB = ((0 * C + 2) * H + y) * W + x
                floatPtr[idxR] = rf
                floatPtr[idxG] = gf
                floatPtr[idxB] = bf
            } else {
                // shape [C, H, W]
                let idxR = (0 * H + y) * W + x
                let idxG = (1 * H + y) * W + x
                let idxB = (2 * H + y) * W + x
                floatPtr[idxR] = rf
                floatPtr[idxG] = gf
                floatPtr[idxB] = bf
            }
        }
    }

    return mlArray
}
