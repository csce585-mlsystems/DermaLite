import SwiftUI
import PhotosUI
import SwiftData
import UIKit

struct ScanView: View {
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var selectedImage: UIImage? = nil          // raw selected or camera image
    @State private var croppedImage: UIImage? = nil           // final image used for analysis
    @State private var cameraImage: UIImage? = nil
    @State private var showCropView = false
    @State private var showCamera = false
    @State private var isAnalyzing = false
    @State private var savedLesion: Lesion? = nil
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                if let image = croppedImage {
                    // Preview of final image to be analyzed
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .shadow(radius: 4)
                        .padding(.horizontal)

                    HStack(spacing: 12) {
                        Button {
                            // allow re-cropping / selecting different area
                            showCropView = true
                        } label: {
                            Label("Select Different Area", systemImage: "crop")
                        }
                        .buttonStyle(.bordered)

                        Button {
                            // Use the full (uncropped) selected image instead of cropped preview
                            if let sel = selectedImage {
                                croppedImage = sel
                            }
                        } label: {
                            Label("Use Full Image", systemImage: "square")
                        }
                        .buttonStyle(.bordered)
                    }

                    Button(action: continueWithImage) {
                        HStack {
                            if isAnalyzing {
                                ProgressView()
                                    .scaleEffect(0.8)
                            }
                            Text(isAnalyzing ? "Analyzing..." : "Continue")
                                .bold()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isAnalyzing)
                    .padding(.top, 6)

                } else {
                    VStack(spacing: 12) {
                        HStack(spacing: 12) {
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

                            PhotosPicker(selection: $selectedItem, matching: .images) {
                                HStack {
                                    Image(systemName: "photo.on.rectangle")
                                    Text("Choose from Library")
                                }
                                .padding(10)
                                .background(RoundedRectangle(cornerRadius: 8).strokeBorder())
                            }
                            .buttonStyle(.plain)
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
                        // When crop completes, set the croppedImage so preview and continue button appear.
                        croppedImage = cropped
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
                    // open crop view immediately so user can refine area
                    showCamera = false
                    showCropView = true
                }
            }
            .onChange(of: selectedItem) { newItem in
                Task {
                    if let data = try? await newItem?.loadTransferable(type: Data.self),
                       let uiImage = UIImage(data: data) {
                        selectedImage = uiImage
                        // let user crop the selection
                        showCropView = true
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
        guard let uiImage = croppedImage ?? selectedImage else { return }
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
                        croppedImage = nil
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
