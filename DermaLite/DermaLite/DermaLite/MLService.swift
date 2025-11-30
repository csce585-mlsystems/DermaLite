// MLService.swift
import Foundation
import CoreML
import UIKit
import Vision
import CoreImage
import Accelerate

class MLService {
    static let shared = MLService()

    // VN wrappers only if model accepts Image input
    private var binaryModel: VNCoreMLModel?
    private var multiclassModel: VNCoreMLModel?

    // Always keep the underlying MLModel for direct multiArray calls
    private var binaryCoreMLModel: MLModel?
    private var multiclassCoreMLModel: MLModel?

    private var classLabels: [String]? = nil
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
        let config = MLModelConfiguration()
        #if targetEnvironment(simulator)
        config.computeUnits = .cpuOnly
        #else
        config.computeUnits = .all
        #endif

        // --- BINARY MODEL ---
        do {
            let binaryWrapper = try dermalite_binary_classifier(configuration: config)
            let binaryML = binaryWrapper.model
            self.binaryCoreMLModel = binaryML

            if let firstInput = binaryML.modelDescription.inputDescriptionsByName.first?.value {
                print("Binary model input type: \(firstInput.type)")
                if firstInput.type == .image {
                    self.binaryModel = try VNCoreMLModel(for: binaryML)
                    print("MLService: Binary VNCoreMLModel created (image input).")
                } else {
                    self.binaryModel = nil
                    print("MLService: Binary model expects non-image input (e.g. multiArray). Will use MLModel.prediction path.")
                }
            } else {
                print("MLService: Binary model has no input description.")
            }
        } catch {
            print("MLService: Failed to load binary classifier - \(error)")
        }

        print("MLService: Binary classifier loaded successfully")

        // --- MULTICLASS MODEL ---
        do {
            let multiclassWrapper = try MobileNetV2_CAM(configuration: config)
            let multiML = multiclassWrapper.model
            self.multiclassCoreMLModel = multiML

            // Extract class labels if available for fallback mapping
            if let labelsAny = multiML.modelDescription.classLabels {
                if let labels = labelsAny as? [String] {
                    self.classLabels = labels
                    print("MLService: Loaded class labels (\(labels.count))")
                } else if let labelsNums = labelsAny as? [NSNumber] {
                    self.classLabels = labelsNums.map { $0.stringValue }
                    print("MLService: Loaded numeric class labels (\(labelsNums.count))")
                } else {
                    print("MLService: classLabels present but unexpected type: \(type(of: labelsAny))")
                }
            } else {
                print("MLService: No class labels found in model description")
            }

            if let firstInput = multiML.modelDescription.inputDescriptionsByName.first?.value {
                print("Multiclass model input type: \(firstInput.type)")
                if firstInput.type == .image {
                    self.multiclassModel = try VNCoreMLModel(for: multiML)
                    print("MLService: Multiclass VNCoreMLModel created (image input).")
                } else {
                    self.multiclassModel = nil
                    print("MLService: Multiclass model expects non-image input (e.g. multiArray). Will use MLModel.prediction path.")
                }
            } else {
                print("MLService: Multiclass model has no input description.")
            }

            print("MLService: Multiclass classifier loaded successfully")
        } catch {
            print("Failed to load multiclass model: \(error)")
        }
    }

    // MARK: - Two-Stage Classification
    func predict(image: UIImage, completion: @escaping (String?, Double?) -> Void) {
        // Stage 1: Run binary classifier first
        runBinaryClassification(image: image) { [weak self] isMalignant, binaryConfidence in
            guard let self = self else {
                DispatchQueue.main.async { completion(nil, nil) }
                return
            }

            // Stage 2: If malignant, run multiclass classifier for specific diagnosis
            print("MLService: Binary classifier determined MALIGNANT (confidence: \(binaryConfidence ?? 0.0)), running multiclass classifier...")
            self.runMulticlassClassification(image: image) { diagnosis, multiclassConfidence in
                DispatchQueue.main.async { completion(diagnosis, multiclassConfidence) }
            }
        }
    }

    // MARK: - Binary Classification
    private func runBinaryClassification(image: UIImage, completion: @escaping (Bool, Double?) -> Void) {
        // If VN wrapper available and model expects image input, use Vision path
        if let vnModel = self.binaryModel {
            let request = VNCoreMLRequest(model: vnModel) { request, error in
                if let error = error {
                    print("MLService Binary: Error - \(error)")
                    completion(false, nil)
                    return
                }

                // Try standard classification results first
                if let results = request.results as? [VNClassificationObservation],
                   let topResult = results.first {
                    print("MLService Binary: Top result = \(topResult.identifier), confidence = \(topResult.confidence)")
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
            return
        }

        // Else: use direct MLModel prediction path (multiArray expected)
        guard let underlyingModel = self.binaryCoreMLModel else {
            print("MLService: Binary CoreML model not available")
            completion(false, nil)
            return
        }

        do {
            let provider = try featureProviderForModel(underlyingModel, image: image)
            let output = try underlyingModel.prediction(from: provider)

            // Try dictionary output first
            for name in output.featureNames {
                if let fv = output.featureValue(for: name) {
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
                        print("MLService Binary (multiArray): Class 0 = \(class0Prob), Class 1 = \(class1Prob)")
                        completion(isMalignant, confidence)
                        return
                    }

                    if let array = fv.multiArrayValue {
                        let vals = self.doubleValues(from: array)
                        let probs = self.softmax(vals)
                        if probs.count >= 2 {
                            let isMalignant = probs[1] > probs[0]
                            let confidence = max(probs[0], probs[1])
                            print("MLService Binary (multiArray): Softmax probs = \(probs), malignant = \(isMalignant)")
                            completion(isMalignant, confidence)
                            return
                        }
                    }
                }
            }

            print("MLService Binary (multiArray): No usable outputs")
            completion(false, nil)
        } catch {
            print("MLService Binary (multiArray): Prediction failed - \(error)")
            completion(false, nil)
        }
    }

    // MARK: - Multiclass Classification
    private func runMulticlassClassification(image: UIImage, completion: @escaping (String?, Double?) -> Void) {
        // If VN wrapper available and model accepts images, use Vision path
        if let vnModel = self.multiclassModel {
            let request = VNCoreMLRequest(model: vnModel) { request, error in
                if let error = error {
                    print("MLService Multiclass: Error - \(error)")
                    completion(nil, nil)
                    return
                }

                // Try standard classification results first
                if let results = request.results as? [VNClassificationObservation],
                   let topResult = results.first {
                    print("MLService Multiclass: Top classification = \(topResult.identifier), confidence = \(topResult.confidence)")

                    let label = topResult.identifier
                    if self.malignantTypes.contains(label) {
                        let diagnosis = self.lesionTypes[label] ?? label
                        completion(diagnosis, Double(topResult.confidence))
                    } else {
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
            return
        }

        // Else: use direct MLModel prediction (multiArray expected)
        guard let underlyingModel = self.multiclassCoreMLModel else {
            print("MLService: Multiclass CoreML model not available")
            completion(nil, nil)
            return
        }

        do {
            let provider = try featureProviderForModel(underlyingModel, image: image)
            let output = try underlyingModel.prediction(from: provider)

            // Try dictionary format first
            for name in output.featureNames {
                if let fv = output.featureValue(for: name) {
                    let dict = fv.dictionaryValue
                    if !dict.isEmpty {
                        var bestLabel: String?
                        var bestProb: Double = 0
                        for (key, value) in dict {
                            guard let label = key as? String else { continue }
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
                            print("MLService Multiclass (multiArray): Best malignant label = \(bestLabel), prob = \(bestProb)")
                            completion(diagnosis, bestProb)
                            return
                        }
                    }

                    if let array = fv.multiArrayValue {
                        let vals = self.doubleValues(from: array)
                        let probs = self.softmax(vals)

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
                            print("MLService Multiclass (multiArray): Softmax best malignant = \(label), prob = \(bestMalignantProb)")
                            completion(diagnosis, bestMalignantProb)
                            return
                        }
                    }
                }
            }

            print("MLService Multiclass (multiArray): No usable malignant results, returning generic malignant")
            completion("Malignant (Requires Medical Evaluation)", nil)
            return
        } catch {
            print("MLService Multiclass (multiArray): Prediction failed - \(error)")
            completion("Malignant (Requires Medical Evaluation)", nil)
            return
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

    // MARK: - NEW: Build MLFeatureProvider for models that expect multiArray inputs
    // This function inspects the model input description and constructs either a pixelBuffer-based provider
    // or a multiArray provider (the latter by converting the UIImage into a normalized MLMultiArray).
    private func featureProviderForModel(_ model: MLModel, image: UIImage, targetSize: CGSize = CGSize(width: 224, height: 224)) throws -> MLFeatureProvider {
        guard let inputDescPair = model.modelDescription.inputDescriptionsByName.first else {
            throw NSError(domain: "MLService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model has no inputs"])
        }
        let inputName = inputDescPair.key
        let desc = inputDescPair.value

        if desc.type == .image {
            // build pixel buffer
            let width = Int(targetSize.width), height = Int(targetSize.height)
            guard let pb = image.toCVPixelBuffer(width: width, height: height) else {
                throw NSError(domain: "MLService", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"])
            }
            let fv = MLFeatureValue(pixelBuffer: pb)
            return try MLDictionaryFeatureProvider(dictionary: [inputName: fv])
        }

        if desc.type == .multiArray {
            // inspect shape constraint if present
            if let constraint = desc.multiArrayConstraint {
                let shape = constraint.shape.map { $0.intValue }
                let batchFirst = (shape.count == 4 && shape[0] == 1)
                // heuristics to extract width/height from shape; fallback to 224
                var h = 224, w = 224
                if shape.count == 4 { h = shape[2]; w = shape[3] }
                else if shape.count == 3 { h = shape[1]; w = shape[2] }

                guard let mlarr = imageToMLMultiArray(image, width: w, height: h, batchFirst: batchFirst) else {
                    throw NSError(domain: "MLService", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to construct MLMultiArray"])
                }
                let fv = MLFeatureValue(multiArray: mlarr)
                return try MLDictionaryFeatureProvider(dictionary: [inputName: fv])
            } else {
                // no constraint info — use default
                guard let mlarr = imageToMLMultiArray(image) else {
                    throw NSError(domain: "MLService", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create default MLMultiArray"])
                }
                let fv = MLFeatureValue(multiArray: mlarr)
                return try MLDictionaryFeatureProvider(dictionary: [inputName: fv])
            }
        }

        // fallback: try pixel buffer
        if let pb = image.toCVPixelBuffer(width: Int(targetSize.width), height: Int(targetSize.height)) {
            let fv = MLFeatureValue(pixelBuffer: pb)
            return try MLDictionaryFeatureProvider(dictionary: [inputName: fv])
        }

        throw NSError(domain: "MLService", code: 5, userInfo: [NSLocalizedDescriptionKey: "Unsupported model input type \(desc.type)"])
    }

    // Convert UIImage -> MLMultiArray (Float32) normalized with ImageNet mean/std.
    // Default output shape = [1,3,H,W] (batchFirst = true). Adjust if model expects different ordering.
    private func imageToMLMultiArray(_ image: UIImage,
                                     width: Int = 224,
                                     height: Int = 224,
                                     mean: (Float,Float,Float) = (0.485, 0.456, 0.406),
                                     std: (Float,Float,Float)  = (0.229, 0.224, 0.225),
                                     batchFirst: Bool = true) -> MLMultiArray? {
        guard let resized = image.resized(to: CGSize(width: width, height: height)),
              let cgImage = resized.cgImage else { return nil }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = 4 * width
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
        guard let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: colorSpace, bitmapInfo: bitmapInfo),
              let data = ctx.data else { return nil }

        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Build MLMultiArray
        let shape: [NSNumber] = batchFirst ? [1, 3, NSNumber(value: height), NSNumber(value: width)] : [3, NSNumber(value: height), NSNumber(value: width)]
        guard let mlArray = try? MLMultiArray(shape: shape, dataType: .float32) else { return nil }

        let ptr = data.bindMemory(to: UInt8.self, capacity: bytesPerRow * height)
        let count = mlArray.count
        let floatPtr = mlArray.dataPointer.bindMemory(to: Float.self, capacity: count)

        let C = 3
        let H = height
        let W = width

        let meanR = mean.0, meanG = mean.1, meanB = mean.2
        let stdR = std.0, stdG = std.1, stdB = std.2

        for y in 0..<H {
            for x in 0..<W {
                let pixelIndex = y * bytesPerRow + x * 4
                let r = ptr[pixelIndex + 1]
                let g = ptr[pixelIndex + 2]
                let b = ptr[pixelIndex + 3]

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

        return mlArray
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

