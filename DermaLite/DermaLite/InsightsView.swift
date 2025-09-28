import SwiftUI

struct InsightsView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 56))
                .foregroundStyle(.secondary)
            Text("Insights")
                .font(.title2)
                .bold()
            Text("Trends and tips will appear here as you add more scans.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
    }
}

#Preview {
    NavigationStack { InsightsView() }
}
