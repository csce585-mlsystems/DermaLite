import SwiftUI

struct LesionDetailView: View {
    let lesion: Lesion
    @State private var isAnalyzing = false

    // NEW: overlay state
    @State private var overlayImage: UIImage? = nil
    @State private var showOverlay: Bool = true
    @State private var overlayOpacity: Double = 0.8
    @State private var isLoadingOverlay: Bool = false

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Image Section (with overlay)
                if let uiImage = loadImage(named: lesion.imageFileName) {
                    ZStack {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 300)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .shadow(radius: 4)

                        if isLoadingOverlay {
                            // subtle loader over image while overlay loads
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle())
                                .scaleEffect(1.2)
                                .padding()
                                .background(.thinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }

                        if let overlay = overlayImage, showOverlay {
                            Image(uiImage: overlay)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 300)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .opacity(overlayOpacity)
                                .allowsHitTesting(false)
                        }
                    }
                }

                // Overlay Controls
                if overlayImage != nil {
                    VStack(spacing: 12) {
                        HStack {
                            Toggle(isOn: $showOverlay) {
                                Text("Show overlay")
                                    .font(.subheadline)
                            }
                            .toggleStyle(SwitchToggleStyle(tint: .accentColor))
                            Spacer()
                        }
                        HStack {
                            Text("Overlay opacity")
                                .font(.subheadline)
                            Slider(value: $overlayOpacity, in: 0.0...1.0)
                                .frame(maxWidth: 200)
                            Text("\(Int(overlayOpacity * 100))%")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .frame(width: 44, alignment: .trailing)
                        }
                    }
                    .padding()
                    .background(Color(.systemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .shadow(radius: 1)
                }

                // Analysis Results Section
                VStack(spacing: 16) {
                    if isAnalyzing {
                        VStack(spacing: 12) {
                            ProgressView()
                                .scaleEffect(1.2)
                            Text("Analyzing...")
                                .font(.headline)
                                .foregroundStyle(.secondary)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else if let diagnosis = lesion.predictedDiagnosis,
                              let confidence = lesion.confidence {
                        // Results Card
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Analysis Results")
                                .font(.title2)
                                .fontWeight(.semibold)

                            VStack(alignment: .leading, spacing: 8) {
                                Text("Predicted Diagnosis")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                Text(diagnosis)
                                    .font(.title3)
                                    .fontWeight(.medium)
                            }

                            VStack(alignment: .leading, spacing: 8) {
                                Text("Confidence")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                HStack {
                                    Text("\(Int(confidence * 100))%")
                                        .font(.title3)
                                        .fontWeight(.medium)
                                    Spacer()
                                    ProgressView(value: confidence)
                                        .frame(width: 100)
                                }
                            }

                            Divider()

                            Text("Note: This is an AI prediction for informational purposes only. Please consult a healthcare professional for medical advice.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding(.top, 4)
                        }
                        .padding()
                        .background(Color(.systemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .shadow(radius: 2)
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "brain.head.profile")
                                .font(.system(size: 40))
                                .foregroundStyle(.secondary)
                            Text("No analysis available")
                                .font(.headline)
                                .foregroundStyle(.secondary)
                            Text("The image was saved before AI analysis was implemented.")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .multilineTextAlignment(.center)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                }

                // Metadata Section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Details")
                        .font(.title2)
                        .fontWeight(.semibold)

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Scan Date")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Text(lesion.createdAt, style: .date)
                            .font(.body)
                        Text(lesion.createdAt, style: .time)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    if let notes = lesion.notes, !notes.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Notes")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            Text(notes)
                                .font(.body)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(.systemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .shadow(radius: 2)

                Spacer()
            }
            .padding()
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Scan Details")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadOverlayAsync()
        }
    }

    // Loads the main saved image from Documents (unchanged)
    private func loadImage(named: String) -> UIImage? {
        let fm = FileManager.default
        if let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            let url = docs.appendingPathComponent(named)
            if let data = try? Data(contentsOf: url) {
                return UIImage(data: data)
            }
        }
        return nil
    }

    // NEW: async overlay loader
    private func loadOverlayAsync() {
        // Don't reload if we already have it
        if overlayImage != nil { return }

        isLoadingOverlay = true
        Task {
            // 1) Prefer explicit overlayFileName saved on the lesion
            if let overlayName = lesion.overlayFileName {
                if let img = loadImage(named: overlayName) {
                    await MainActor.run {
                        self.overlayImage = img
                        self.isLoadingOverlay = false
                    }
                    return
                }
            }

            // 2) Fallback deterministic naming (if you saved overlay as "<imageFileName>-overlay.jpg")
            let fallback = fallbackOverlayName(for: lesion.imageFileName)
            if let img = loadImage(named: fallback) {
                await MainActor.run {
                    self.overlayImage = img
                    self.isLoadingOverlay = false
                }
                return
            }

            // 3) If overlay not found, attempt to generate on-device (optional)
            // If you want to trigger CAM generation here, call your CAM generation API:
            // e.g. Task.detached { let overlay = CAMGenerator.shared.generateCAMOverlay(for: croppedUIImage); save to documents; update lesion.overlayFileName }
            // We skip generation here to keep the view loading-only.

            await MainActor.run {
                self.isLoadingOverlay = false
            }
        }
    }

    // Helper that returns deterministic overlay filename used by ScanView when overlay was saved by convention
    private func fallbackOverlayName(for imageFileName: String) -> String {
        // If imageFileName is "abcd.jpg" -> "abcd-overlay.jpg"
        let ext = (imageFileName as NSString).pathExtension
        let base = (imageFileName as NSString).deletingPathExtension
        let suffix = "\(base)-overlay"
        return ext.isEmpty ? suffix : "\(suffix).\(ext)"
    }
}
