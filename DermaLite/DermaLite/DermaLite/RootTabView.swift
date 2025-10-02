import SwiftUI

struct RootTabView: View {
    @State private var selectedTab: Tab = .scan

    enum Tab: Hashable {
        case scan
        case library
        case insights
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                ScanView()
                    .navigationTitle("Scan")
            }
            .tabItem {
                Label("Scan", systemImage: "camera.viewfinder")
            }
            .tag(Tab.scan)

            NavigationStack {
                LibraryView()
                    .navigationTitle("Library")
            }
            .tabItem {
                Label("Library", systemImage: "square.grid.2x2")
            }
            .tag(Tab.library)

            NavigationStack {
                InsightsView()
                    .navigationTitle("Insights")
            }
            .tabItem {
                Label("Insights", systemImage: "chart.bar.xaxis")
            }
            .tag(Tab.insights)
        }
    }
}

#Preview {
    RootTabView()
}
