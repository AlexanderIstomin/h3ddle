import AppKit
import H3ddleGeneration
import SwiftUI

@main
struct H3ddleApp: App {
  @State private var model = AppModel(
    generationProvider: ProcessInfo.processInfo.arguments.contains("-H3ddleFastFakeGeneration")
      ? FakeGenerationProvider(stepDelay: .milliseconds(20))
      : FakeGenerationProvider()
  )

  var body: some Scene {
    WindowGroup("H3ddle") {
      EditorView(model: model)
        .frame(minWidth: 1_080, minHeight: 700)
        .preferredColorScheme(.dark)
        .background(WindowChromeInstaller())
        .onReceive(
          NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)
        ) { _ in
          DockAttention.clear()
        }
        .onReceive(
          NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)
        ) { _ in
          DockAttention.clear()
          model.shutdownEngine()
        }
    }
    .defaultSize(width: 1_280, height: 820)
    .commands {
      CommandGroup(replacing: .newItem) {}
    }
  }
}
