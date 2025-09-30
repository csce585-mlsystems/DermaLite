import SwiftUI

struct LesionDetailView: View {
    let lesion: Lesion
    @State private var isAnalyzing = false

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Image Section
                if let uiImage = loadImage(named: lesion.imageFileName) {
                    Image(uiImage: uiImage)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 300)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .shadow(radius: 4)
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
    }

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
}