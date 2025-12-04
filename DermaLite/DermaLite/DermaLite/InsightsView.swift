import SwiftUI
import SwiftData
import Charts

private struct ScanPerDay: Identifiable {
    let date: Date
    let count: Int
    var id: Date { date }
}

private struct DiagnosisBreakdown: Identifiable {
    let label: String
    let count: Int
    var id: String { label }
}

struct InsightsView: View {
    @Query(sort: \Lesion.createdAt, order: .forward) private var lesions: [Lesion]
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Summary statistics
                                HStack {
                    VStack(alignment: .leading) {
                        Text("Total Scans")
                            .font(.caption).foregroundStyle(.secondary)
                        Text("\(lesions.count)")
                            .font(.title2).bold()
                    }
                    Spacer()
                    VStack(alignment: .leading) {
                        Text("Unique Diagnoses")
                            .font(.caption).foregroundStyle(.secondary)
                        Text("\(Set(lesions.compactMap { $0.predictedDiagnosis }).count)")
                            .font(.title2).bold()
                    }
                }
                .padding(.bottom, 8)

                Divider()

                // Bar chart: Scans per day
                if !lesions.isEmpty {
                    Text("Scans Per Day")
                        .font(.headline)
                    Chart(scansPerDay(lesions: lesions)) { item in
                        BarMark(
                            x: .value("Date", item.date, unit: .day),
                            y: .value("Scans", item.count)
                        )
                    }
                    .frame(height: 160)
                }

                // Pie chart: Diagnosis breakdown
                if !lesions.isEmpty {
                    Text("Diagnosis Breakdown")
                        .font(.headline)
                    Chart(diagnosisBreakdown(lesions: lesions)) { item in
                        SectorMark(
                            angle: .value("Count", item.count),
                            innerRadius: .ratio(0.55),
                            angularInset: 2
                        )
                        .foregroundStyle(by: .value("Diagnosis", item.label))
                    }
                    .frame(height: 200)
                    .chartLegend(position: .trailing)
                }

                if lesions.isEmpty {
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
                }
            }
            .padding()
        }
        .background(Color(.systemGroupedBackground))
    }
}

private func scansPerDay(lesions: [Lesion]) -> [ScanPerDay] {
    let calendar = Calendar.current
    let grouped = Dictionary(grouping: lesions) { calendar.startOfDay(for: $0.createdAt) }
    return grouped.map { ScanPerDay(date: $0.key, count: $0.value.count) }
        .sorted { $0.date < $1.date }
}

private func diagnosisBreakdown(lesions: [Lesion]) -> [DiagnosisBreakdown] {
    let grouped = Dictionary(grouping: lesions.compactMap { $0.predictedDiagnosis }) { $0 }
    return grouped.map { DiagnosisBreakdown(label: $0.key, count: $0.value.count) }
        .sorted { $0.count > $1.count }
}

#Preview {
    NavigationStack { InsightsView() }
}
