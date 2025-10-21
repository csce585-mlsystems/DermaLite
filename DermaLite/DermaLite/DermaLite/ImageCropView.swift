import SwiftUI

struct ImageCropView: View {
    let originalImage: UIImage
    let onCropComplete: (UIImage) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var cropBoxPosition: CGPoint = .zero
    @State private var imageSize: CGSize = .zero
    @State private var displayScale: CGFloat = 1.0
    @State private var croppedPreview: UIImage?
    @State private var showPreview = false
    @State private var cropSizeInPixels: CGFloat = 400 // Uniform starting size for all images

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 20) {
                Text("Position and resize the box over the lesion")
                    .font(.headline)
                    .padding(.top)

                // Main crop area
                ZStack {
                    // Display the image
                    Image(uiImage: orientedImage)
                        .resizable()
                        .scaledToFit()
                        .background(
                            GeometryReader { imgGeo in
                                Color.clear
                                    .onAppear {
                                        setupImageGeometry(imgGeo.size, containerSize: geometry.size)
                                    }
                            }
                        )

                    // Overlay with draggable crop box
                    if imageSize != .zero {
                        CropBoxOverlay(
                            cropBoxPosition: $cropBoxPosition,
                            cropBoxSize: cropBoxSize,
                            cropSizeInPixels: $cropSizeInPixels,
                            displayScale: displayScale,
                            imageSize: imageSize,
                            maxCropSize: min(orientedImage.size.width, orientedImage.size.height),
                            onPositionChange: updatePreview
                        )
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: geometry.size.height * 0.6)

                // Preview section
                VStack(spacing: 12) {
                    Text("Preview (\(Int(cropSizeInPixels))×\(Int(cropSizeInPixels)))")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    if let preview = croppedPreview {
                        Image(uiImage: preview)
                            .resizable()
                            .scaledToFit()
                            .frame(width: 120, height: 120)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.blue, lineWidth: 2)
                            )
                    } else {
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.gray.opacity(0.2))
                            .frame(width: 120, height: 120)
                            .overlay(
                                Text("Preview")
                                    .foregroundStyle(.secondary)
                            )
                    }
                }

                // Action buttons
                HStack(spacing: 16) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .buttonStyle(.bordered)

                    Button("Confirm Selection") {
                        if let cropped = cropImage() {
                            onCropComplete(cropped)
                            dismiss()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(croppedPreview == nil)
                }
                .padding(.bottom)
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .navigationTitle("Crop Lesion")
    }

    // MARK: - Computed Properties

    /// Get properly oriented image (ensure vertical)
    private var orientedImage: UIImage {
        guard let cgImage = originalImage.cgImage else { return originalImage }

        // Check if image needs rotation
        let width = cgImage.width
        let height = cgImage.height

        // If width > height, rotate to make it vertical
        if width > height {
            return rotateImage(originalImage, by: .pi / 2) ?? originalImage
        }

        return originalImage
    }

    /// Calculate the size of the crop box on screen (proportional to displayed image)
    private var cropBoxSize: CGSize {
        guard imageSize != .zero else { return .zero }

        // The crop box should represent 244x244 pixels of the actual image
        // Scale it proportionally to the displayed image size
        let boxSize = cropSizeInPixels * displayScale
        return CGSize(width: boxSize, height: boxSize)
    }

    // MARK: - Helper Functions

    private func setupImageGeometry(_ displaySize: CGSize, containerSize: CGSize) {
        imageSize = displaySize

        // Calculate scale: how many screen points = 1 image pixel
        let actualImageSize = orientedImage.size
        displayScale = displaySize.width / actualImageSize.width

        // Set initial crop size to 400px (uniform starting size)
        cropSizeInPixels = 400

        // Center the crop box initially
        cropBoxPosition = CGPoint(
            x: displaySize.width / 2,
            y: displaySize.height / 2
        )

        // Generate initial preview
        updatePreview()
    }

    private func updatePreview() {
        croppedPreview = cropImage()
    }

    private func cropImage() -> UIImage? {
        guard imageSize != .zero else { return nil }

        let actualImageSize = orientedImage.size

        // Convert crop box position from screen coordinates to image coordinates
        let scaleX = actualImageSize.width / imageSize.width
        let scaleY = actualImageSize.height / imageSize.height

        // Calculate crop rect in image coordinates
        let cropX = (cropBoxPosition.x - cropBoxSize.width / 2) * scaleX
        let cropY = (cropBoxPosition.y - cropBoxSize.height / 2) * scaleY

        let cropRect = CGRect(
            x: max(0, cropX),
            y: max(0, cropY),
            width: cropSizeInPixels,
            height: cropSizeInPixels
        )

        // Perform the crop
        guard let cgImage = orientedImage.cgImage,
              let croppedCGImage = cgImage.cropping(to: cropRect) else {
            return nil
        }

        return UIImage(cgImage: croppedCGImage, scale: orientedImage.scale, orientation: orientedImage.imageOrientation)
    }

    private func rotateImage(_ image: UIImage, by radians: CGFloat) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        let rotatedSize = CGSize(width: cgImage.height, height: cgImage.width)

        UIGraphicsBeginImageContextWithOptions(rotatedSize, false, image.scale)
        guard let context = UIGraphicsGetCurrentContext() else { return nil }

        context.translateBy(x: rotatedSize.width / 2, y: rotatedSize.height / 2)
        context.rotate(by: radians)
        context.translateBy(x: -image.size.width / 2, y: -image.size.height / 2)

        image.draw(at: .zero)

        let rotatedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return rotatedImage
    }
}

// MARK: - Crop Box Overlay

struct CropBoxOverlay: View {
    @Binding var cropBoxPosition: CGPoint
    let cropBoxSize: CGSize
    @Binding var cropSizeInPixels: CGFloat
    let displayScale: CGFloat
    let imageSize: CGSize
    let maxCropSize: CGFloat
    let onPositionChange: () -> Void

    @State private var dragOffset: CGSize = .zero
    @State private var baseCropSize: CGFloat = 0

    var body: some View {
        ZStack {
            // Semi-transparent overlay
            Rectangle()
                .fill(Color.black.opacity(0.5))
                .mask {
                    Rectangle()
                        .overlay {
                            Rectangle()
                                .frame(width: cropBoxSize.width, height: cropBoxSize.height)
                                .position(cropBoxPosition)
                                .blendMode(.destinationOut)
                        }
                }

            // Crop box
            Rectangle()
                .strokeBorder(Color.white, lineWidth: 2)
                .background(Color.clear)
                .frame(width: cropBoxSize.width, height: cropBoxSize.height)
                .position(cropBoxPosition)
                .overlay(
                    // Corner indicators
                    ZStack {
                        ForEach(0..<4) { index in
                            CornerIndicator()
                                .position(cornerPosition(index))
                        }
                    }
                )
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            let newX = cropBoxPosition.x + value.translation.width - dragOffset.width
                            let newY = cropBoxPosition.y + value.translation.height - dragOffset.height

                            // Constrain to image bounds
                            let halfBox = cropBoxSize.width / 2
                            let clampedX = min(max(newX, halfBox), imageSize.width - halfBox)
                            let clampedY = min(max(newY, halfBox), imageSize.height - halfBox)

                            cropBoxPosition = CGPoint(x: clampedX, y: clampedY)
                            dragOffset = value.translation
                        }
                        .onEnded { _ in
                            dragOffset = .zero
                            onPositionChange()
                        }
                )
                .simultaneousGesture(
                    MagnificationGesture()
                        .onChanged { value in
                            // Store base size when gesture starts
                            if baseCropSize == 0 {
                                baseCropSize = cropSizeInPixels
                            }

                            let newSize = baseCropSize * value

                            // Constrain size: minimum 100px, maximum is smallest image dimension
                            let clampedSize = min(max(newSize, 100), maxCropSize)
                            cropSizeInPixels = clampedSize

                            // Ensure crop box stays within bounds after resize
                            let halfBox = clampedSize * displayScale / 2
                            let clampedX = min(max(cropBoxPosition.x, halfBox), imageSize.width - halfBox)
                            let clampedY = min(max(cropBoxPosition.y, halfBox), imageSize.height - halfBox)
                            cropBoxPosition = CGPoint(x: clampedX, y: clampedY)
                        }
                        .onEnded { _ in
                            baseCropSize = 0
                            onPositionChange()
                        }
                )

            // Center crosshair
            Path { path in
                // Horizontal line
                path.move(to: CGPoint(x: cropBoxPosition.x - 10, y: cropBoxPosition.y))
                path.addLine(to: CGPoint(x: cropBoxPosition.x + 10, y: cropBoxPosition.y))
                // Vertical line
                path.move(to: CGPoint(x: cropBoxPosition.x, y: cropBoxPosition.y - 10))
                path.addLine(to: CGPoint(x: cropBoxPosition.x, y: cropBoxPosition.y + 10))
            }
            .stroke(Color.white, lineWidth: 1)
        }
        .frame(width: imageSize.width, height: imageSize.height)
    }

    private func cornerPosition(_ index: Int) -> CGPoint {
        let halfWidth = cropBoxSize.width / 2
        let halfHeight = cropBoxSize.height / 2

        switch index {
        case 0: // Top-left
            return CGPoint(x: cropBoxPosition.x - halfWidth, y: cropBoxPosition.y - halfHeight)
        case 1: // Top-right
            return CGPoint(x: cropBoxPosition.x + halfWidth, y: cropBoxPosition.y - halfHeight)
        case 2: // Bottom-left
            return CGPoint(x: cropBoxPosition.x - halfWidth, y: cropBoxPosition.y + halfHeight)
        case 3: // Bottom-right
            return CGPoint(x: cropBoxPosition.x + halfWidth, y: cropBoxPosition.y + halfHeight)
        default:
            return cropBoxPosition
        }
    }
}

struct CornerIndicator: View {
    var body: some View {
        Rectangle()
            .fill(Color.white)
            .frame(width: 20, height: 3)
            .overlay(
                Rectangle()
                    .fill(Color.white)
                    .frame(width: 3, height: 20)
            )
    }
}

#Preview {
    NavigationStack {
        ImageCropView(originalImage: UIImage(systemName: "photo")!) { _ in }
    }
}
