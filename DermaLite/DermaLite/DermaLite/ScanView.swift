import SwiftUI
import PhotosUI
import SwiftData
import UIKit

struct ScanView: View {
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var selectedImage: UIImage? = nil          // selected or camera image (may be cropped)
    @State private var cameraImage: UIImage? = nil
    @State private var showCropView = false
    @State private var showCamera = false
    @State private var isAnalyzing = false
    @State private var savedLesion: Lesion? = nil
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                if let image = selectedImage {
                    // Preview image (show selectedImage, which may be original or cropped)
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .shadow(radius: 4)
                        .padding(.horizontal)

                    // New 3-button layout
                    VStack(spacing: 12) {
                        Button {
                            showCropView = true
                        } label: {
                            Label("Crop", systemImage: "crop")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .frame(maxWidth: .infinity)

                        Button {
                            // Cancel - reset and return to scan view
                            selectedImage = nil
                            selectedItem = nil
                            cameraImage = nil
                        } label: {
                            Label("Cancel", systemImage: "xmark")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .frame(maxWidth: .infinity)

                        Button(action: continueWithImage) {
                            if isAnalyzing {
                                HStack {
                                    ProgressView()
                                        .scaleEffect(0.8)
                                    Text("Analyzing...")
                                        .bold()
                                }
                                .frame(maxWidth: .infinity)
                            } else {
                                Label("Confirm", systemImage: "checkmark")
                                    .frame(maxWidth: .infinity)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .frame(maxWidth: .infinity)
                        .disabled(isAnalyzing)
                    }
                    .padding(.horizontal)

                } else {
                    VStack(spacing: 12) {
                        VStack(spacing: 12) {
                            Button {
                                cameraImage = nil
                                showCamera = true
                            } label: {
                                HStack {
                                    Image(systemName: "camera")
                                    Text("Scan (Camera)")
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .frame(maxWidth: .infinity)

                            PhotosPicker(selection: $selectedItem, matching: .images) {
                                HStack {
                                    Image(systemName: "photo.on.rectangle")
                                    Text("Choose from Library")
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .frame(maxWidth: .infinity)
                        }
                        Text("Tip: After choosing a photo you can crop it before analyzing.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.top, 6)
                    }
                    .padding(.horizontal)
                }
            }
            .navigationDestination(isPresented: $showCropView) {
                if let image = selectedImage {
                    ImageCropView(originalImage: image) { cropped in
                        // Update selectedImage with cropped version
                        selectedImage = cropped
                        showCropView = false  // Return to preview
                    }
                } else {
                    // fallback UI (shouldn't usually happen)
                    Text("No image to crop")
                        .font(.headline)
                        .foregroundStyle(.secondary)
                }
            }
            .sheet(isPresented: $showCamera) {
                ImagePicker(sourceType: .camera, selectedImage: $cameraImage)
                    .ignoresSafeArea()
            }
            .onChange(of: cameraImage) { newImage in
                if let img = newImage {
                    selectedImage = img
                    showCamera = false
                    // Don't auto-open crop view - let user choose
                }
            }
            .onChange(of: selectedItem) { newItem in
                Task {
                    if let data = try? await newItem?.loadTransferable(type: Data.self),
                       let uiImage = UIImage(data: data) {
                        selectedImage = uiImage
                        // Don't auto-open crop view - let user choose
                    }
                }
            }
            .navigationDestination(item: $savedLesion) { lesion in
                LesionDetailView(lesion: lesion)
            }
            .onAppear {
                // ensure a fresh scan UI when returning to this tab
                selectedItem = nil
                cameraImage = nil
                // do not automatically clear selectedImage/croppedImage here so user can return
            }
        }
    }

    // MARK: - Actions

    private func continueWithImage() {
        guard let uiImage = selectedImage else { return }
        isAnalyzing = true

        Task { @MainActor in
            do {
                let fileName = UUID().uuidString + ".jpg"
                _ = try saveImageToDocuments(uiImage: uiImage, fileName: fileName)

                // Create lesion and save
                let lesion = Lesion(imageFileName: fileName)
                modelContext.insert(lesion)
                try modelContext.save()

                // Run ML inference
                MLService.shared.predict(image: uiImage) { diagnosis, confidence in
                    Task { @MainActor in
                        lesion.predictedDiagnosis = diagnosis
                        lesion.confidence = confidence
                        try? modelContext.save()

                        // Trigger navigation to detail view
                        savedLesion = lesion
                        isAnalyzing = false

                        // Reset scan UI
                        selectedItem = nil
                        selectedImage = nil
                        cameraImage = nil
                        showCropView = false
                        showCamera = false
                    }
                }
            } catch {
                print("Failed to save image: \(error)")
                isAnalyzing = false
            }
        }
    }

    // MARK: - Helpers
    private func saveImageToDocuments(uiImage: UIImage, fileName: String) throws -> URL {
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

// MARK: - ImagePicker (UIKit wrapper)
struct ImagePicker: UIViewControllerRepresentable {
    enum SourceType {
        case camera
        case photoLibrary

        var uiImagePickerSource: UIImagePickerController.SourceType {
            switch self {
            case .camera: return .camera
            case .photoLibrary: return .photoLibrary
            }
        }
    }

    var sourceType: SourceType = .photoLibrary
    @Binding var selectedImage: UIImage?

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = sourceType.uiImagePickerSource
        picker.allowsEditing = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker
        init(_ parent: ImagePicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            picker.dismiss(animated: true)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}
