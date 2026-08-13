import AppKit
import SwiftUI

@main
struct H3ddleApp: App {
  @State private var model = AppModel()

  var body: some Scene {
    WindowGroup("H3ddle") {
      EditorView(model: model)
        .frame(minWidth: 1_080, minHeight: 700)
        .preferredColorScheme(.dark)
        .onReceive(
          NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)
        ) { _ in
          model.shutdownEngine()
        }
    }
    .windowStyle(.hiddenTitleBar)
    .defaultSize(width: 1_280, height: 820)
    .commands {
      CommandGroup(replacing: .newItem) {}
    }
  }
}
