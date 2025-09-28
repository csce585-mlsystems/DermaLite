import SwiftUI
import SwiftData

struct LibraryView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Lesion.createdAt, order: .reverse) private var lesions: [Lesion]

    var body: some View {
        Group {
            if lesions.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "square.grid.2x2")
                        .font(.system(size: 56))
                        .foregroundStyle(.secondary)
                    Text("Your Library")
                        .font(.title2)
                        .bold()
                    Text("Saved scans will appear here. Use the Scan tab to add a new entry.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
            } else {
                List(lesions) { lesion in
                    HStack(spacing: 12) {
                        ThumbnailView(fileName: lesion.imageFileName)
                            .frame(width: 64, height: 64)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        VStack(alignment: .leading, spacing: 4) {
                            Text(lesion.createdAt, style: .date)
                                .font(.headline)
                            Text(lesion.createdAt, style: .time)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .listStyle(.insetGrouped)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if !lesions.isEmpty {
                    Button(role: .destructive) {
                        for lesion in lesions { modelContext.delete(lesion) }
                        try? modelContext.save()
                    } label: {
                        Image(systemName: "trash")
                    }
                }
            }
        }
    }
}

struct ThumbnailView: View {
    let fileName: String
    var body: some View {
        if let uiImage = loadImage(named: fileName) {
            Image(uiImage: uiImage)
                .resizable()
                .scaledToFill()
        } else {
            ZStack {
                Color.secondary.opacity(0.15)
                Image(systemName: "photo")
                    .foregroundStyle(.secondary)
            }
        }
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

#Preview {
    NavigationStack { LibraryView() }
}
