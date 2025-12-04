import Foundation
import UIKit
import CoreML
import Accelerate
import CoreImage

final class CAMGenerator {
    static let shared = CAMGenerator()

    private var modelURL: URL?
    private var model: MLModel?
    private var classifierWeights: [Float] = []
    private var numClasses: Int = 0
    private var channels: Int = 0

    // Name of model resource in bundle (adjust if needed)
    private let modelResourceName = "MobileNetV2_CAM"
    private let classifierWeightsResource = "classifier_weights" // .bin (flat float32) or .json

    private init() {
        loadModel()
        loadClassifierWeights()
    }

    // MARK: - Public
    /// Generate CAM overlay for `input` using the bundled Core ML model and classifier weights.
    /// Returns composited UIImage or nil (and prints diagnostics).
    func generateCAMOverlay(for input: UIImage) -> UIImage? {
        guard let model = model else {
            print("CAMGenerator: MLModel not loaded")
            return nil
        }

        // Build feature provider automatically for this model
        let inputDescPair = model.modelDescription.inputDescriptionsByName.first
        guard let (inputName, inputDesc) = inputDescPair else {
            print("CAMGenerator: model has no input descriptions")
            dumpModelSchema(model)
            return nil
        }

        // Create provider
        let provider: MLFeatureProvider
        do {
            provider = try featureProviderForModel(model, inputImage: input, targetSize: preferredInputSize(from: inputDesc))
        } catch {
            print("CAMGenerator: failed to build feature provider - \(error.localizedDescription)")
            dumpModelSchema(model)
            return nil
        }

        // Run prediction with retry for Espresso/MPSGraph errors
        let output: MLFeatureProvider
        do {
            output = try predictWithRetry(model: model, provider: provider)
        } catch {
            print("CAMGenerator: model prediction failed: \(error)")
            return nil
        }

        // Find logits (class probabilities) and activations
        guard let (logitsName, logitsArray) = firstMultiArrayLike(in: output, preferLength: nil),
              let logits = logitsArray else {
            print("CAMGenerator: could not find logits output in model outputs. Available outputs: \(output.featureNames)")
            dumpFeatureProvider(output)
            return nil
        }

        // For activations, search outputs for something that looks spatial (C,H,W or 1,C,H,W)
        guard let activationName = findActivationOutputName(in: model, from: output) else {
            print("CAMGenerator: could not detect activations output automatically. Outputs: \(model.modelDescription.outputDescriptionsByName.keys)")
            return nil
        }

        guard let activationValue = output.featureValue(for: activationName)?.multiArrayValue else {
            print("CAMGenerator: activations output '\(activationName)' exists but is not MLMultiArray or nil")
            return nil
        }

        // --- DEBUG: Print MLMultiArray shape at runtime ---
        let actShape = activationValue.shape.map { $0.intValue }
        print("[DEBUG] CAM activation MLMultiArray shape: \(actShape)")
        print("[DEBUG] CAM activation MLMultiArray: dataType=\(activationValue.dataType), count=\(activationValue.count)")

        // Convert activations to float buffer and compute CAM
        let (featChannels, featH, featW, activationsFlat) = multiArrayToCHW(activationValue)
        self.channels = featChannels

        // Determine predicted class index from logits
        let logitsFloats = logits.toFloatArray()
        print("logits shape:", logits.shape)
        print("logits count:", logits.count)
        guard let predIndex = logitsFloats.enumerated().max(by: { $0.element < $1.element })?.offset else {
            print("CAMGenerator: cannot determine predicted class from logits")
            return nil
        }
        if predIndex >= numClasses {
            print("CAMGenerator: predIndex (\(predIndex)) is out of range for numClasses (\(numClasses))")
            return nil
        }

        // Ensure classifier weights are loaded and have expected shape
        if classifierWeights.isEmpty || numClasses == 0 || classifierWeights.count != numClasses * channels {
            print("CAMGenerator: classifier weights mismatch: weights.count=\(classifierWeights.count), numClasses=\(numClasses), channels=\(channels)")
            return nil
        }

        // compute CAM: heatmap = ReLU(sum_c w_c * activation_c)
        var heatmap = [Float](repeating: 0, count: featH * featW)
        let weightBase = predIndex * channels
        for c in 0..<channels {
            print("weightBase:", weightBase, "channels:", channels, "classifierWeights.count:", classifierWeights.count)
            let w = classifierWeights[weightBase + c]
            let base = c * featH * featW
            for i in 0..<(featH * featW) {
                heatmap[i] += w * activationsFlat[base + i]
            }
        }

        // ReLU + normalize
        vDSP_vthr(heatmap, 1, [0.0], &heatmap, 1, vDSP_Length(heatmap.count))
        var maxv: Float = 0
        vDSP_maxv(heatmap, 1, &maxv, vDSP_Length(heatmap.count))
        if maxv > 0 {
            var inv = 1.0 / maxv
            vDSP_vsmul(heatmap, 1, &inv, &heatmap, 1, vDSP_Length(heatmap.count))
        }

        // Resize heatmap to input image size and colorize
        guard let heatUIImage = heatmapToUIImage(heatmap, width: featW, height: featH, targetSize: input.size) else {
            print("CAMGenerator: failed to convert heatmap to UIImage")
            return nil
        }

        // Composite: blend heatmap (use overlay alpha 0.45 by default)
        let overlayed = blend(original: input, overlay: heatUIImage, alpha: 0.45)
        return overlayed
    }

    // MARK: - Model / weights loading
    private func loadModel() {
        guard let url = Bundle.main.url(forResource: modelResourceName, withExtension: "mlpackage") ??
                        Bundle.main.url(forResource: modelResourceName, withExtension: "mlmodelc") ??
                        Bundle.main.url(forResource: modelResourceName, withExtension: "mlmodel") else {
            print("CAMGenerator: model '\(modelResourceName)' not found in bundle")
            return
        }
        self.modelURL = url

        // Try load with default computeUnits (.all). If it fails with Espresso/MpsGraph message, we'll retry later on prediction.
        do {
            let config = MLModelConfiguration()
            #if targetEnvironment(simulator)
            config.computeUnits = .cpuOnly
            #else
            config.computeUnits = .all
            #endif
            self.model = try MLModel(contentsOf: url, configuration: config)
            print("CAMGenerator: loaded MLModel from \(url.lastPathComponent) with computeUnits=\(config.computeUnits)")
            dumpModelSchema(self.model!)
        } catch {
            // Will try with CPU only as fallback
            print("CAMGenerator: failed to load MLModel with .all: \(error). Trying cpuOnly...")
            do {
                let cfg = MLModelConfiguration()
                cfg.computeUnits = .cpuOnly
                self.model = try MLModel(contentsOf: url, configuration: cfg)
                print("CAMGenerator: loaded MLModel with cpuOnly")
                dumpModelSchema(self.model!)
            } catch {
                print("CAMGenerator: failed to load model even with cpuOnly: \(error)")
            }
        }
    }

    private func loadClassifierWeights() {
        // Try binary file (.bin float32) first
        if let url = Bundle.main.url(forResource: classifierWeightsResource, withExtension: "bin"),
           let data = try? Data(contentsOf: url) {
            let expectedCount = data.count / MemoryLayout<Float>.size
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                let floatPtr = raw.bindMemory(to: Float.self)
                classifierWeights = Array(floatPtr)
            }
            // We cannot know numClasses or channels reliably from file alone; leave user to set or infer later
            // But we can attempt to guess numClasses by checking model outputs
            if let m = model, let out = m.modelDescription.outputDescriptionsByName.first?.value,
               let multi = out.multiArrayConstraint {
                // if logits available as multi-array, we can set numClasses accordingly
                // This is heuristic: assume the first multi-array-shaped output is logits vector
                for (name, desc) in m.modelDescription.outputDescriptionsByName {
                    if let c = desc.multiArrayConstraint {
                        let count = c.shape.reduce(1) { $0 * $1.intValue }
                        numClasses = count
                        break
                    }
                }
            }
            print("CAMGenerator: loaded classifier_weights.bin count=\(classifierWeights.count), guessed numClasses=\(numClasses)")
            return
        }

        // Try JSON (array of arrays)
        if let url = Bundle.main.url(forResource: classifierWeightsResource, withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) {
            if let arr = json as? [[Double]] {
                numClasses = arr.count
                var flat = [Float]()
                for row in arr {
                    for v in row { flat.append(Float(v)) }
                }
                classifierWeights = flat
                print("CAMGenerator: loaded classifier_weights.json rows=\(numClasses), weights=\(classifierWeights.count)")
                return
            } else if let flat = json as? [Double] {
                classifierWeights = flat.map { Float($0) }
                print("CAMGenerator: loaded flat classifier weights JSON count=\(classifierWeights.count)")
                return
            }
        }

        print("CAMGenerator: classifier weights not found in bundle (tried .bin and .json). CAM will not run.")
    }

    // MARK: - Prediction with retry on Espresso/MPSGraph errors
    private func predictWithRetry(model: MLModel, provider: MLFeatureProvider) throws -> MLFeatureProvider {
        do {
            return try model.prediction(from: provider)
        } catch {
            let errMsg = String(describing: error)
            if errMsg.contains("MpsGraph") || errMsg.contains("Espresso") || errMsg.contains("MPSGraph") {
                // Retry with cpuOnly
                print("CAMGenerator: detected Espresso/MPSGraph runtime error; retrying with cpuOnly")
                guard let url = modelURL else { throw error }
                let cfg = MLModelConfiguration()
                cfg.computeUnits = .cpuOnly
                let cpuModel = try MLModel(contentsOf: url, configuration: cfg)
                self.model = cpuModel
                return try cpuModel.prediction(from: provider)
            } else {
                throw error
            }
        }
    }

    // MARK: - Helpers: detect activation output name heuristically
    private func findActivationOutputName(in mlModel: MLModel, from outputProvider: MLFeatureProvider) -> String? {
        // Prefer outputs with names containing activ/feature/map
        for name in mlModel.modelDescription.outputDescriptionsByName.keys {
            let lower = name.lowercased()
            if lower.contains("activ") || lower.contains("feature") || lower.contains("map") || lower.contains("conv") {
                return name
            }
        }

        // Otherwise pick the first multi-array output whose dimensionality suggests spatial dims (>=3)
        for (name, desc) in mlModel.modelDescription.outputDescriptionsByName {
            if let constraint = desc.multiArrayConstraint {
                let shape = constraint.shape.map { $0.intValue }
                // Accept shapes like [1, C, H, W] or [C, H, W]
                if shape.count >= 3 {
                    return name
                }
            }
        }

        // As last resort, try provider feature names for multiArray with dims >1
        for name in outputProvider.featureNames {
            if let fv = outputProvider.featureValue(for: name), fv.type == .multiArray, let arr = fv.multiArrayValue {
                if arr.shape.count >= 3 { return name }
            }
        }

        return nil
    }

    // Helper that returns name or nil
    private func firstMultiArrayLike(in provider: MLFeatureProvider, preferLength: Int?) -> (String, MLMultiArray?)? {
        for name in provider.featureNames {
            if let fv = provider.featureValue(for: name) {
                if let arr = fv.multiArrayValue {
                    // if preferLength provided, try to match length
                    if let pl = preferLength {
                        if arr.count == pl {
                            return (name, arr)
                        }
                    } else {
                        return (name, arr)
                    }
                }
            }
        }
        return nil
    }

    // Convert MLMultiArray activations to (C,H,W) flatten layout and return array
    private func multiArrayToCHW(_ arr: MLMultiArray) -> (Int, Int, Int, [Float]) {
        let shape = arr.shape.map { $0.intValue }
        var C = 0, H = 0, W = 0
        var baseOffset = 0
        if shape.count == 4 {
            // [1, C, H, W]
            C = shape[1]; H = shape[2]; W = shape[3]
            baseOffset = 0
        } else if shape.count == 3 {
            // [C, H, W]
            C = shape[0]; H = shape[1]; W = shape[2]
            baseOffset = 0
        } else if shape.count == 2 {
            // treat as [C, N] fallback
            C = shape[0]; H = 1; W = shape[1]
        } else {
            // flatten fallback
            let total = arr.count
            return (1, 1, total, arr.toFloatArray())
        }

        let flat = arr.toFloatArray()
        // If array is already in C,H,W in row-major with channels grouped, we expect flat layout as [c0 elements, c1 elements,...]
        return (C, H, W, flat)
    }

    // Convert heatmap floats to grayscale UIImage and resize to target size with colorization
    private func heatmapToUIImage(_ heatmap: [Float], width: Int, height: Int, targetSize: CGSize) -> UIImage? {
        let count = width * height
        guard heatmap.count >= count else { return nil }

        // Apply Jet colormap (or Turbo, or similar)
        // We'll construct an RGBA pixel buffer
        var rgbaPixels = [UInt8](repeating: 0, count: count * 4)
        for i in 0..<count {
            let v = max(0, min(1, heatmap[i])) // Clamp to 0...1
            let (r, g, b) = jetColormap(v)
            rgbaPixels[i * 4 + 0] = UInt8(r * 255)
            rgbaPixels[i * 4 + 1] = UInt8(g * 255)
            rgbaPixels[i * 4 + 2] = UInt8(b * 255)
            rgbaPixels[i * 4 + 3] = UInt8((v > 0.05 ? 255 : 0)) // use some threshold for transparency
        }

        let data = Data(rgbaPixels)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        guard let cgImage = CGImage(width: width,
                                    height: height,
                                    bitsPerComponent: 8,
                                    bitsPerPixel: 32,
                                    bytesPerRow: width * 4,
                                    space: CGColorSpaceCreateDeviceRGB(),
                                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                    provider: provider,
                                    decode: nil,
                                    shouldInterpolate: true,
                                    intent: .defaultIntent) else { return nil }
        let uiImage = UIImage(cgImage: cgImage)

        // Resize colormap image to match the target input image size
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 0)
        uiImage.draw(in: CGRect(origin: .zero, size: targetSize))
        let result = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return result
    }

    // Jet colormap: input is 0...1, output is (r,g,b) floats 0...1
    private func jetColormap(_ value: Float) -> (Float, Float, Float) {
        let v = max(0, min(1, value))
        let fourValue = 4 * v
        let r = min(fourValue - 1.5, -fourValue + 4.5)
        let g = min(fourValue - 0.5, -fourValue + 3.5)
        let b = min(fourValue + 0.5, -fourValue + 2.5)
        return (clamp01(r), clamp01(g), clamp01(b))
    }

    private func clamp01(_ x: Float) -> Float {
        return min(max(x, 0), 1)
    }

    // Simple blend
    private func blend(original: UIImage, overlay: UIImage, alpha: CGFloat) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(original.size, false, original.scale)
        original.draw(in: CGRect(origin: .zero, size: original.size))
        overlay.draw(in: CGRect(origin: .zero, size: original.size), blendMode: .normal, alpha: alpha)
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return out
    }

    // MARK: - Utility: build feature provider from UIImage (image or MLMultiArray)
    private func featureProviderForModel(_ model: MLModel, inputImage: UIImage, targetSize: CGSize) throws -> MLFeatureProvider {
        guard let (inputName, inputDesc) = model.modelDescription.inputDescriptionsByName.first else {
            throw NSError(domain: "CAMGenerator", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model has no inputs"])
        }

        if inputDesc.type == .image {
            let width = Int(targetSize.width), height = Int(targetSize.height)
            guard let pb = inputImage.toCVPixelBuffer(width: width, height: height) else {
                throw NSError(domain: "CAMGenerator", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create CVPixelBuffer"])
            }
            return try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pb)])
        } else if inputDesc.type == .multiArray {
            var batchFirst = true
            if let c = inputDesc.multiArrayConstraint {
                let shape = c.shape.map { $0.intValue }
                batchFirst = (shape.count == 4 && shape[0] == 1)
                var w = 224, h = 224
                if shape.count == 4 { h = shape[2]; w = shape[3] }
                else if shape.count == 3 { h = shape[1]; w = shape[2] }
                guard let arr = imageToMLMultiArray(inputImage, width: w, height: h, batchFirst: batchFirst) else {
                    throw NSError(domain: "CAMGenerator", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed building MLMultiArray"])
                }
                return try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
            } else {
                // fallback
                guard let arr = imageToMLMultiArray(inputImage) else {
                    throw NSError(domain: "CAMGenerator", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed building default MLMultiArray"])
                }
                return try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
            }
        } else {
            // Fallback to CVPixelBuffer
            guard let pb = inputImage.toCVPixelBuffer(width: Int(targetSize.width), height: Int(targetSize.height)) else {
                throw NSError(domain: "CAMGenerator", code: 5, userInfo: [NSLocalizedDescriptionKey: "Unsupported model input type and failed pb fallback"])
            }
            return try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pb)])
        }
    }

    // Build MLMultiArray normalized to ImageNet mean/std, default shape [1,3,H,W]
    private func imageToMLMultiArray(_ image: UIImage, width: Int = 224, height: Int = 224, mean: (Float,Float,Float) = (0.485,0.456,0.406), std: (Float,Float,Float) = (0.229,0.224,0.225), batchFirst: Bool = true) -> MLMultiArray? {
        guard let resized = image.resized(to: CGSize(width: width, height: height)), let cg = resized.cgImage else { return nil }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = 4 * width
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo), let data = ctx.data else {
            return nil
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))

        let shape: [NSNumber] = batchFirst ? [1, 3, NSNumber(value: height), NSNumber(value: width)] : [3, NSNumber(value: height), NSNumber(value: width)]
        guard let arr = try? MLMultiArray(shape: shape, dataType: .float32) else { return nil }
        let ptr = data.bindMemory(to: UInt8.self, capacity: bytesPerRow * height)
        let floatPtr = arr.dataPointer.bindMemory(to: Float.self, capacity: arr.count)
        let C = 3, H = height, W = width
        let meanR = mean.0, meanG = mean.1, meanB = mean.2
        let stdR = std.0, stdG = std.1, stdB = std.2

        for y in 0..<H {
            for x in 0..<W {
                let pixelIndex = y * bytesPerRow + x * 4
                let r = ptr[pixelIndex + 1], g = ptr[pixelIndex + 2], b = ptr[pixelIndex + 3]
                let rf = (Float(r) / 255.0 - meanR) / stdR
                let gf = (Float(g) / 255.0 - meanG) / stdG
                let bf = (Float(b) / 255.0 - meanB) / stdB
                if batchFirst {
                    let idxR = ((0 * C + 0) * H + y) * W + x
                    let idxG = ((0 * C + 1) * H + y) * W + x
                    let idxB = ((0 * C + 2) * H + y) * W + x
                    floatPtr[idxR] = rf
                    floatPtr[idxG] = gf
                    floatPtr[idxB] = bf
                } else {
                    let idxR = (0 * H + y) * W + x
                    let idxG = (1 * H + y) * W + x
                    let idxB = (2 * H + y) * W + x
                    floatPtr[idxR] = rf
                    floatPtr[idxG] = gf
                    floatPtr[idxB] = bf
                }
            }
        }
        return arr
    }

    // Utilities for logging
    private func dumpModelSchema(_ model: MLModel) {
        print("CAMGenerator: MODEL SCHEMA INPUTS:")
        for (k,v) in model.modelDescription.inputDescriptionsByName { print("  INPUT: \(k) -> \(v.type), multiArrayConstraint=\(String(describing: v.multiArrayConstraint))") }
        print("CAMGenerator: MODEL SCHEMA OUTPUTS:")
        for (k,v) in model.modelDescription.outputDescriptionsByName { print("  OUTPUT: \(k) -> \(v.type), multiArrayConstraint=\(String(describing: v.multiArrayConstraint))") }
    }

    private func dumpFeatureProvider(_ provider: MLFeatureProvider) {
        print("CAMGenerator: provider features: \(provider.featureNames)")
    }
    
    // Helper to determine input image size from MLFeatureDescription or fallback to 224x224
    private func preferredInputSize(from desc: MLFeatureDescription) -> CGSize {
        if let c = desc.imageConstraint {
            return CGSize(width: c.pixelsWide, height: c.pixelsHigh)
        } else if let c = desc.multiArrayConstraint {
            let shape = c.shape.map { $0.intValue }
            if shape.count >= 3 {
                // [C,H,W] or [1,C,H,W]
                let h = shape[shape.count - 2]
                let w = shape[shape.count - 1]
                return CGSize(width: w, height: h)
            }
        }
        // Default fallback
        return CGSize(width: 224, height: 224)
    }
}

// MARK: - Helper extensions
fileprivate extension MLMultiArray {
    func toFloatArray() -> [Float] {
        let cnt = self.count
        var arr = [Float](repeating: 0, count: cnt)
        switch self.dataType {
        case .float32:
            let ptr = self.dataPointer.bindMemory(to: Float.self, capacity: cnt)
            for i in 0..<cnt { arr[i] = ptr[i] }
        case .double, .float64:
            let ptr = self.dataPointer.bindMemory(to: Double.self, capacity: cnt)
            for i in 0..<cnt { arr[i] = Float(ptr[i]) }
        case .float16:
            let ptr = self.dataPointer.bindMemory(to: UInt16.self, capacity: cnt)
            for i in 0..<cnt { arr[i] = Float(Float16(bitPattern: ptr[i])) }
        default:
            for i in 0..<cnt { arr[i] = Float(truncating: self[i]) }
        }
        return arr
    }
}

// MARK: - Local UIImage extension helpers (duplicate of MLInputHelpers)
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
