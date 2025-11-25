import Foundation
import CoreML
import UIKit
import Vision
import CoreImage

class MLService {
    static let shared = MLService()

    // Binary classifier (benign vs malignant)
    private var binaryModel: VNCoreMLModel?

    // Multiclass classifier (specific malignancy types)
    private var multiclassModel: VNCoreMLModel?
    private var classLabels: [String]? = nil

    // Mole detector (Stage 0 - pre-filter)
    private var moleDetectorModel: VNCoreMLModel?
    private let classLabelsFallback: [String] = [
        "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
    ]

    private let lesionTypes = [
        "akiec": "Actinic Keratoses",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevi",
        "vasc": "Vascular Lesions"
    ]

    // Malignant lesion types (for filtering multiclass results)
    private let malignantTypes: Set<String> = ["akiec", "bcc", "mel"]

    private init() {
        loadModels()
    }

    private func loadModels() {
        do {
            let config = MLModelConfiguration()
            #if targetEnvironment(simulator)
            config.computeUnits = .cpuOnly
            #else
            config.computeUnits = .all
            #endif

            // Load mole detector (Stage 0)
            let moleDetectorCoreMLModel = try mole_detector(configuration: config)
            moleDetectorModel = try VNCoreMLModel(for: moleDetectorCoreMLModel.model)
            print("MLService: Mole detector loaded successfully")

            // Load binary classifier
            let binaryCoreMLModel = try dermalite_binary_classifier(configuration: config)
            binaryModel = try VNCoreMLModel(for: binaryCoreMLModel.model)
            print("MLService: Binary classifier loaded successfully")

            // Load multiclass classifier
            let multiclassCoreMLModel = try dermalite_mobilenetv2(configuration: config)
            multiclassModel = try VNCoreMLModel(for: multiclassCoreMLModel.model)
            
            // Extract class labels if available for fallback mapping
            let labelsAny = multiclassCoreMLModel.model.modelDescription.classLabels
            if let labels = labelsAny as? [String] {
                self.classLabels = labels
                print("MLService: Loaded class labels (\(labels.count))")
            } else if let labelsNums = labelsAny as? [NSNumber] {
                let labels = labelsNums.map { $0.stringValue }
                self.classLabels = labels
                print("MLService: Loaded numeric class labels (\(labels.count))")
            } else {
                self.classLabels = nil
                print("MLService: No class labels found in model description")
            }

            print("MLService: Multiclass classifier loaded successfully")
        } catch {
            print("Failed to load models: \(error)")
        }
    }

    // MARK: - Three-Stage Classification
    func predict(image: UIImage, completion: @escaping (String?, Double?) -> Void) {
        // Stage 0: Run mole detector first
        runMoleDetection(image: image) { [weak self] isMole, moleConfidence in
            guard let self = self else {
                DispatchQueue.main.async { completion(nil, nil) }
                return
            }

            // Early exit if no mole detected
            if !isMole {
                // Invert confidence for display: if model says 10% chance of mole,
                // we want to show 90% confidence it's NOT a mole
                let notMoleConfidence = moleConfidence.map { 1.0 - $0 }
                print("MLService: Mole detector determined NO MOLE (confidence: \(notMoleConfidence ?? 0.0))")
                DispatchQueue.main.async {
                    completion("No Mole Detected", notMoleConfidence)
                }
                return
            }

            print("MLService: Mole detector determined MOLE PRESENT (confidence: \(moleConfidence ?? 0.0)), running binary classifier...")

            // Stage 1: Run binary classifier
            self.runBinaryClassification(image: image) { isMalignant, binaryConfidence in
                // If benign, return immediately with binary result
                if !isMalignant {
                    print("MLService: Binary classifier determined BENIGN (confidence: \(binaryConfidence ?? 0.0))")
                    DispatchQueue.main.async { completion("Benign", binaryConfidence) }
                    return
                }

                // Stage 2: If malignant, run multiclass classifier for specific diagnosis
                print("MLService: Binary classifier determined MALIGNANT (confidence: \(binaryConfidence ?? 0.0)), running multiclass classifier...")
                self.runMulticlassClassification(image: image) { diagnosis, multiclassConfidence in
                    DispatchQueue.main.async { completion(diagnosis, multiclassConfidence) }
                }
            }
        }
    }

    // MARK: - Binary Classification
    private func runBinaryClassification(image: UIImage, completion: @escaping (Bool, Double?) -> Void) {
        guard let binaryModel = binaryModel else {
            print("MLService: Binary model not loaded")
            completion(false, nil)
            return
        }

        let request = VNCoreMLRequest(model: binaryModel) { request, error in
            if let error = error {
                print("MLService Binary: Error - \(error)")
                completion(false, nil)
                return
            }

            // Try standard classification results first
            if let results = request.results as? [VNClassificationObservation],
               let topResult = results.first {
                print("MLService Binary: Top result = \(topResult.identifier), confidence = \(topResult.confidence)")
                // Class 0 = Benign, Class 1 = Malignant
                let isMalignant = (topResult.identifier == "1")
                completion(isMalignant, Double(topResult.confidence))
                return
            }

            // Fallback: handle feature value observations
            if let featureObservation = request.results?.first as? VNCoreMLFeatureValueObservation {
                let fv = featureObservation.featureValue

                // Try dictionary format
                let dict = fv.dictionaryValue
                if !dict.isEmpty {
                    var class0Prob: Double = 0.0
                    var class1Prob: Double = 0.0
                    for (key, value) in dict {
                        if let label = key as? String {
                            let prob = value.doubleValue
                            if label == "0" { class0Prob = prob }
                            else if label == "1" { class1Prob = prob }
                        }
                    }
                    let isMalignant = class1Prob > class0Prob
                    let confidence = max(class0Prob, class1Prob)
                    print("MLService Binary: Class 0 (benign) = \(class0Prob), Class 1 (malignant) = \(class1Prob)")
                    completion(isMalignant, confidence)
                    return
                }

                // Try multi-array format
                if let array = fv.multiArrayValue {
                    let vals = self.doubleValues(from: array)
                    let probs = self.softmax(vals)
                    if probs.count >= 2 {
                        let isMalignant = probs[1] > probs[0]
                        let confidence = max(probs[0], probs[1])
                        print("MLService Binary: Softmax probs = \(probs), malignant = \(isMalignant)")
                        completion(isMalignant, confidence)
                        return
                    }
                }
            }

            print("MLService Binary: No usable results")
            completion(false, nil)
        }

        #if targetEnvironment(simulator)
        request.usesCPUOnly = true
        #endif
        request.imageCropAndScaleOption = .centerCrop

        performVisionRequest(request, on: image) { success in
            if !success {
                completion(false, nil)
            }
        }
    }

    // MARK: - Multiclass Classification
    private func runMulticlassClassification(image: UIImage, completion: @escaping (String?, Double?) -> Void) {
        guard let multiclassModel = multiclassModel else {
            print("MLService: Multiclass model not loaded")
            completion(nil, nil)
            return
        }

        let request = VNCoreMLRequest(model: multiclassModel) { request, error in
            if let error = error {
                print("MLService Multiclass: Error - \(error)")
                completion(nil, nil)
                return
            }

            // Try standard classification results first
            if let results = request.results as? [VNClassificationObservation],
               let topResult = results.first {
                print("MLService Multiclass: Top classification = \(topResult.identifier), confidence = \(topResult.confidence)")

                // Filter to only malignant types (trust binary classifier)
                let label = topResult.identifier
                if self.malignantTypes.contains(label) {
                    let diagnosis = self.lesionTypes[label] ?? label
                    completion(diagnosis, Double(topResult.confidence))
                } else {
                    // Binary said malignant but multiclass returned benign type
                    // Trust binary classifier - return generic "Malignant" warning
                    print("MLService Multiclass: Conflict - multiclass returned benign type '\(label)', but binary classifier said malignant. Returning generic malignant.")
                    completion("Malignant (Requires Medical Evaluation)", Double(topResult.confidence))
                }
                return
            }

            // Fallback: handle feature value observations
            if let featureObservation = request.results?.first as? VNCoreMLFeatureValueObservation {
                let fv = featureObservation.featureValue

                // Try dictionary format
                let dict = fv.dictionaryValue
                if !dict.isEmpty {
                    var bestLabel: String?
                    var bestProb: Double = 0
                    for (key, value) in dict {
                        guard let label = key as? String else { continue }
                        // Only consider malignant types
                        if self.malignantTypes.contains(label) {
                            let prob = value.doubleValue
                            if prob > bestProb {
                                bestProb = prob
                                bestLabel = label
                            }
                        }
                    }
                    if let bestLabel = bestLabel {
                        let diagnosis = self.lesionTypes[bestLabel] ?? bestLabel
                        print("MLService Multiclass: Best malignant label = \(bestLabel), prob = \(bestProb)")
                        completion(diagnosis, bestProb)
                        return
                    }
                }

                // Try multi-array format
                if let array = fv.multiArrayValue {
                    let vals = self.doubleValues(from: array)
                    let probs = self.softmax(vals)

                    // Find best malignant class
                    var bestMalignantIndex: Int?
                    var bestMalignantProb: Double = 0

                    for (index, prob) in probs.enumerated() {
                        let label: String
                        if let labels = self.classLabels, index < labels.count {
                            label = labels[index]
                        } else if index < self.classLabelsFallback.count {
                            label = self.classLabelsFallback[index]
                        } else {
                            continue
                        }

                        if self.malignantTypes.contains(label) && prob > bestMalignantProb {
                            bestMalignantProb = prob
                            bestMalignantIndex = index
                        }
                    }

                    if let index = bestMalignantIndex {
                        let label: String
                        if let labels = self.classLabels, index < labels.count {
                            label = labels[index]
                        } else {
                            label = self.classLabelsFallback[index]
                        }
                        let diagnosis = self.lesionTypes[label] ?? label
                        print("MLService Multiclass: Softmax best malignant = \(label), prob = \(bestMalignantProb)")
                        completion(diagnosis, bestMalignantProb)
                        return
                    }
                }
            }

            print("MLService Multiclass: No usable malignant results, returning generic malignant")
            completion("Malignant (Requires Medical Evaluation)", nil)
        }

        #if targetEnvironment(simulator)
        request.usesCPUOnly = true
        #endif
        request.imageCropAndScaleOption = .centerCrop

        performVisionRequest(request, on: image) { success in
            if !success {
                completion(nil, nil)
            }
        }
    }

    // MARK: - Mole Detection (Stage 0)
    private func runMoleDetection(image: UIImage, completion: @escaping (Bool, Double?) -> Void) {
        guard let moleDetectorModel = moleDetectorModel else {
            print("MLService: Mole detector model not loaded")
            completion(false, nil)
            return
        }

        // Convert UIImage to CVPixelBuffer
        guard let pixelBuffer = convertToCVPixelBuffer(image: image) else {
            print("MLService: Failed to convert image to CVPixelBuffer")
            completion(false, nil)
            return
        }

        // Add Gaussian noise (critical for mole detector robustness)
        guard let noisyPixelBuffer = addGaussianNoise(to: pixelBuffer) else {
            print("MLService: Failed to add Gaussian noise")
            completion(false, nil)
            return
        }

        let request = VNCoreMLRequest(model: moleDetectorModel) { request, error in
            if let error = error {
                print("MLService Mole Detector: Error - \(error)")
                completion(false, nil)
                return
            }

            // Try standard classification results
            if let results = request.results as? [VNClassificationObservation],
               let topResult = results.first {
                print("MLService Mole Detector: Probability = \(topResult.confidence)")
                let isMole = Double(topResult.confidence) >= 0.5
                completion(isMole, Double(topResult.confidence))
                return
            }

            // Fallback: handle feature value observations
            if let featureObservation = request.results?.first as? VNCoreMLFeatureValueObservation {
                let fv = featureObservation.featureValue

                if let array = fv.multiArrayValue {
                    let vals = self.doubleValues(from: array)
                    if let probability = vals.first {
                        print("MLService Mole Detector: Probability = \(probability)")
                        let isMole = probability >= 0.5
                        completion(isMole, probability)
                        return
                    }
                }
            }

            print("MLService Mole Detector: No usable results")
            completion(false, nil)
        }

        #if targetEnvironment(simulator)
        request.usesCPUOnly = true
        #endif
        request.imageCropAndScaleOption = .centerCrop

        // Perform Vision request with noisy pixel buffer
        let handler = VNImageRequestHandler(cvPixelBuffer: noisyPixelBuffer, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                print("MLService Mole Detector: Vision request failed - \(error)")
                completion(false, nil)
            }
        }
    }

    // MARK: - Vision Request Helper
    private func performVisionRequest(_ request: VNCoreMLRequest, on image: UIImage, completion: @escaping (Bool) -> Void) {
        let orientation = CGImagePropertyOrientation(image.imageOrientation)

        let handler: VNImageRequestHandler
        if let cgImage = image.cgImage {
            handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
        } else if let ciImage = image.ciImage {
            handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation, options: [:])
        } else if let ciImage = CIImage(image: image) {
            handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation, options: [:])
        } else {
            completion(false)
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
                completion(true)
            } catch {
                print("MLService: Vision request failed - \(error)")
                completion(false)
            }
        }
    }

    // MARK: - Helpers
    private func doubleValues(from array: MLMultiArray) -> [Double] {
        let count = array.count
        switch array.dataType {
        case .double, .float64:
            let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            return (0..<count).map { ptr[$0] }
        case .float32:
            let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
            return (0..<count).map { Double(ptr[$0]) }
        case .float16:
            let ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Double(Float16(bitPattern: ptr[$0])) }
        default:
            // Fallback: return zeros if unsupported type
            return Array(repeating: 0.0, count: count)
        }
    }

    private func softmax(_ logits: [Double]) -> [Double] {
        guard let maxLogit = logits.max() else { return logits.map { _ in 0 } }
        let exps = logits.map { exp($0 - maxLogit) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else { return logits.map { _ in 0 } }
        return exps.map { $0 / sum }
    }

    // MARK: - CVPixelBuffer Helpers
    private func convertToCVPixelBuffer(image: UIImage) -> CVPixelBuffer? {
        let targetSize = CGSize(width: 224, height: 224)

        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        defer { UIGraphicsEndImageContext() }

        image.draw(in: CGRect(origin: .zero, size: targetSize))
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext(),
              let cgImage = resizedImage.cgImage else {
            return nil
        }

        let width = cgImage.width
        let height = cgImage.height

        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    private func addGaussianNoise(to pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        var noisyPixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            CVPixelBufferGetPixelFormatType(pixelBuffer),
            nil,
            &noisyPixelBuffer
        )

        guard status == kCVReturnSuccess, let outputBuffer = noisyPixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(outputBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(outputBuffer, []) }

        guard let outputAddress = CVPixelBufferGetBaseAddress(outputBuffer) else {
            return nil
        }

        let inputPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        let outputPtr = outputAddress.assumingMemoryBound(to: UInt8.self)

        for row in 0..<height {
            for col in 0..<width {
                let pixelOffset = row * bytesPerRow + col * 4

                // For each RGB channel (BGRA format, skip alpha at offset+3)
                for channel in 0..<3 {
                    let offset = pixelOffset + channel
                    let originalValue = Double(inputPtr[offset])

                    // Add noise: uniform random in [-12.75, 12.75]
                    let noise = Double.random(in: -12.75...12.75)
                    let noisyValue = originalValue + noise

                    // Clamp to [0, 255]
                    let clampedValue = max(0, min(255, noisyValue))
                    outputPtr[offset] = UInt8(clampedValue)
                }

                // Copy alpha channel unchanged
                outputPtr[pixelOffset + 3] = inputPtr[pixelOffset + 3]
            }
        }

        return outputBuffer
    }
}

private extension CGImagePropertyOrientation {
    init(_ uiOrientation: UIImage.Orientation) {
        switch uiOrientation {
        case .up: self = .up
        case .down: self = .down
        case .left: self = .left
        case .right: self = .right
        case .upMirrored: self = .upMirrored
        case .downMirrored: self = .downMirrored
        case .leftMirrored: self = .leftMirrored
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}
