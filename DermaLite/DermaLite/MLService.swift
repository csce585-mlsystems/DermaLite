import Foundation
import CoreML
import UIKit
import Vision

class MLService {
    static let shared = MLService()
    private var model: VNCoreMLModel?

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
        guard let modelURL = Bundle.main.url(forResource: "dermalite_model", withExtension: "mlpackage") else {
            print("Model file not found")
            return
        }

        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            model = try VNCoreMLModel(for: mlModel)
        } catch {
            print("Failed to load model: \(error)")
        }
    }

    func predict(image: UIImage, completion: @escaping (String?, Double?) -> Void) {
        guard let model = model else {
            completion(nil, nil)
            return
        }

        guard let cgImage = image.cgImage else {
            completion(nil, nil)
            return
        }

        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                completion(nil, nil)
                return
            }

            let diagnosis = self.lesionTypes[topResult.identifier] ?? topResult.identifier
            let confidence = Double(topResult.confidence)

            DispatchQueue.main.async {
                completion(diagnosis, confidence)
            }
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async {
                    completion(nil, nil)
                }
            }
        }
    }
}