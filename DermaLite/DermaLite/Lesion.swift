import Foundation
import SwiftData

@Model
final class Lesion {
    @Attribute(.unique) var id: UUID
    var createdAt: Date
    var imageFileName: String
    var notes: String?

    init(id: UUID = UUID(), createdAt: Date = Date(), imageFileName: String, notes: String? = nil) {
        self.id = id
        self.createdAt = createdAt
        self.imageFileName = imageFileName
        self.notes = notes
    }
}
