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

    // NEW: optional overlay filename stored for Grad-CAM / CAM overlays (saved in Documents)
    var overlayFileName: String?

    init(id: UUID = UUID(),
         createdAt: Date = Date(),
         imageFileName: String,
         notes: String? = nil,
         predictedDiagnosis: String? = nil,
         confidence: Double? = nil,
         overlayFileName: String? = nil) {
        self.id = id
        self.createdAt = createdAt
        self.imageFileName = imageFileName
        self.notes = notes
        self.predictedDiagnosis = predictedDiagnosis
        self.confidence = confidence
        self.overlayFileName = overlayFileName
    }
}
