import SwiftUI
import SwiftData

@main
struct DermaLiteApp: App {
    var body: some Scene {
        WindowGroup {
            RootTabView()
        }
        .modelContainer(for: [Lesion.self])
    }
}
