import Foundation
import CoreML
import UIKit
import Vision
import CoreImage

class MLService {
    static let shared = MLService()
    private var model: VNCoreMLModel?
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

    private init() {
        loadModel()
    }

    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            #if targetEnvironment(simulator)
            config.computeUnits = .cpuOnly
            #else
            config.computeUnits = .all
            #endif
            let coreMLModel = try dermalite_model(configuration: config)
            model = try VNCoreMLModel(for: coreMLModel.model)
            
            // Extract class labels if available for fallback mapping
            let labelsAny = coreMLModel.model.modelDescription.classLabels
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
            
            print("MLService: Model loaded successfully via dermalite_model")
        } catch {
            print("Failed to load model: \(error)")
        }
    }

    func predict(image: UIImage, completion: @escaping (String?, Double?) -> Void) {
        guard let model = model else {
            DispatchQueue.main.async { completion(nil, nil) }
            return
        }

        let request = VNCoreMLRequest(model: model) { request, error in
            if let error = error {
                print("MLService: VNCoreMLRequest error: \(error)")
            }
            print("MLService: VN request results count = \(request.results?.count ?? 0)")
            if let first = request.results?.first {
                print("MLService: VN first result type = \(type(of: first))")
            }

            // Try standard classification results first
            if let results = request.results as? [VNClassificationObservation],
               let topResult = results.first {
                print("MLService: Top classification = \(topResult.identifier), confidence = \(topResult.confidence)")
                let diagnosis = self.lesionTypes[topResult.identifier] ?? topResult.identifier
                let confidence = Double(topResult.confidence)
                DispatchQueue.main.async { completion(diagnosis, confidence) }
                return
            }

            // Fallback: handle feature value observations (e.g., probability dictionaries or arrays)
            if let featureObservation = request.results?.first as? VNCoreMLFeatureValueObservation {
                let fv = featureObservation.featureValue

                // Attempt to parse dictionary of label -> probability
                let dict = fv.dictionaryValue
                print("MLService: Feature dictionary entries = \(dict.count)")
                if !dict.isEmpty {
                    var bestLabel: String?
                    var bestProb: Double = 0
                    for (key, value) in dict {
                        guard let label = key as? String else { continue }
                        let prob = value.doubleValue
                        if prob > bestProb {
                            bestProb = prob
                            bestLabel = label
                        }
                    }
                    if let bestLabel = bestLabel {
                        let diagnosis = self.lesionTypes[bestLabel] ?? bestLabel
                        print("MLService: Parsed best label = \(bestLabel), prob = \(bestProb)")
                        DispatchQueue.main.async { completion(diagnosis, bestProb) }
                        return
                    }
                }

                // Attempt to parse multi-array output (e.g., raw logits or probabilities)
                if let array = fv.multiArrayValue {
                    // Read raw values from MLMultiArray with correct data type handling
                    let vals = self.doubleValues(from: array)
                    // Convert logits/scores to probabilities via softmax
                    let probs = self.softmax(vals)
                    // Find top class
                    if let (maxIndex, maxProb) = probs.enumerated().max(by: { $0.element < $1.element }) {
                        let label: String
                        if let labels = self.classLabels, maxIndex < labels.count {
                            label = labels[maxIndex]
                        } else if maxIndex < self.classLabelsFallback.count {
                            label = self.classLabelsFallback[maxIndex]
                        } else {
                            label = String(maxIndex)
                        }
                        let diagnosis = self.lesionTypes[label] ?? label
                        print("MLService: Softmax top index = \(maxIndex), label = \(label), prob = \(maxProb)")
                        DispatchQueue.main.async { completion(diagnosis, maxProb) }
                        return
                    }
                }
            }

            // No usable results
            print("MLService: No usable results from Vision request")
            DispatchQueue.main.async { completion(nil, nil) }
        }

        #if targetEnvironment(simulator)
        request.usesCPUOnly = true
        #endif
        request.imageCropAndScaleOption = .centerCrop

        // Ensure correct orientation and support for CIImage-backed UIImages
        let orientation = CGImagePropertyOrientation(image.imageOrientation)

        let handler: VNImageRequestHandler
        if let cgImage = image.cgImage {
            print("MLService: Using cgImage for VNImageRequestHandler")
            handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
        } else if let ciImage = image.ciImage {
            print("MLService: Using existing ciImage for VNImageRequestHandler")
            handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation, options: [:])
        } else if let ciImage = CIImage(image: image) {
            print("MLService: Created ciImage from UIImage for VNImageRequestHandler")
            handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation, options: [:])
        } else {
            DispatchQueue.main.async { completion(nil, nil) }
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async { completion(nil, nil) }
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
