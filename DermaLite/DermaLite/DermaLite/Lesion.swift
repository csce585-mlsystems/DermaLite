import Foundation
import SwiftData

@Model
final class Lesion {
    @Attribute(.unique) var id: UUID
    var createdAt: Date
    var imageFileName: String
    var notes: String?
    var predictedDiagnosis: String?
    var confidence: Double?

    init(id: UUID = UUID(), createdAt: Date = Date(), imageFileName: String, notes: String? = nil, predictedDiagnosis: String? = nil, confidence: Double? = nil) {
        self.id = id
        self.createdAt = createdAt
        self.imageFileName = imageFileName
        self.notes = notes
        self.predictedDiagnosis = predictedDiagnosis
        self.confidence = confidence
    }
}
