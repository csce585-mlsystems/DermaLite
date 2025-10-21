import SwiftUI
import PhotosUI
import SwiftData

struct ScanView: View {
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var selectedImage: UIImage? = nil
    @State private var croppedImage: UIImage? = nil
    @State private var showCropView = false
    @State private var isAnalyzing = false
    @State private var savedLesion: Lesion? = nil
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        NavigationStack {
            VStack {
                if let image = croppedImage {
                    // Show the cropped image with Continue button
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: 244, height: 244)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.blue, lineWidth: 2)
                        )
                        .padding()

                    Text("Cropped Lesion (\(Int(image.size.width))×\(Int(image.size.width)))")
                        .font(.headline)
                        .padding(.bottom, 8)

                    Button("Continue") {
                        let uiImage = image
                        isAnalyzing = true

                        Task { @MainActor in
                            do {
                                let fileName = UUID().uuidString + ".jpg"
                                let url = try saveImageToDocuments(uiImage: uiImage, fileName: fileName)

                                // Create lesion first
                                let lesion = Lesion(imageFileName: fileName)
                                modelContext.insert(lesion)
                                try modelContext.save()

                                // Perform ML inference
                                MLService.shared.predict(image: uiImage) { diagnosis, confidence in
                                    Task { @MainActor in
                                        lesion.predictedDiagnosis = diagnosis
                                        lesion.confidence = confidence
                                        try? modelContext.save()

                                        savedLesion = lesion
                                        isAnalyzing = false

                                        // Reset selection after saving
                                        selectedItem = nil
                                        selectedImage = nil
                                        croppedImage = nil
                                    }
                                }
                            } catch {
                                print("Failed to save image: \(error)")
                                isAnalyzing = false
                            }
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isAnalyzing)

                    Button("Select Different Area") {
                        croppedImage = nil
                        showCropView = true
                    }
                    .buttonStyle(.bordered)
                    .padding(.top, 8)

                    if isAnalyzing {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text("Analyzing image...")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.top, 8)
                    }
                } else {
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Label("Select Photo", systemImage: "photo.on.rectangle")
                            .font(.headline)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                    .onChange(of: selectedItem) { newItem in
                        Task {
                            if let data = try? await newItem?.loadTransferable(type: Data.self),
                               let uiImage = UIImage(data: data) {
                                selectedImage = uiImage
                                showCropView = true
                            }
                        }
                    }
                }
            }
            .navigationDestination(isPresented: $showCropView) {
                if let image = selectedImage {
                    ImageCropView(originalImage: image) { cropped in
                        croppedImage = cropped
                    }
                }
            }
            .navigationDestination(item: $savedLesion) { lesion in
                LesionDetailView(lesion: lesion)
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
