import SwiftUI
import PhotosUI
import SwiftData

struct ScanView: View {
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var selectedImage: UIImage? = nil
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        VStack {
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                Button("Continue") {
                    let uiImage = image
                    Task { @MainActor in
                        do {
                            let fileName = UUID().uuidString + ".jpg"
                            let url = try saveImageToDocuments(uiImage: uiImage, fileName: fileName)
                            // Insert model
                            let lesion = Lesion(imageFileName: fileName)
                            modelContext.insert(lesion)
                            try modelContext.save()
                            // Reset selection after saving
                            selectedItem = nil
                            selectedImage = nil
                        } catch {
                            print("Failed to save image: \(error)")
                        }
                    }
                }
            } else {
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    Text("Select Photo")
                }
                .onChange(of: selectedItem) { newItem in
                    Task {
                        if let data = try? await newItem?.loadTransferable(type: Data.self),
                           let uiImage = UIImage(data: data) {
                            selectedImage = uiImage
                        }
                    }
                }
            }
        }
    }
}

private extension ScanView {
    func saveImageToDocuments(uiImage: UIImage, fileName: String) throws -> URL {
        guard let data = uiImage.jpegData(compressionQuality: 0.9) else {
            throw NSError(domain: "DermaLite", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to encode JPEG data"])
        }
        let fm = FileManager.default
        let docs = try fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let url = docs.appendingPathComponent(fileName)
        try data.write(to: url, options: .atomic)
        return url
    }
}

#Preview {
    ScanView()
}
