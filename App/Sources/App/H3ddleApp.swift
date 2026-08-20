import AppKit
import H3ddleGeneration
import H3ddleModels
import SwiftUI

@main
struct H3ddleApp: App {
  @State private var model: AppModel = {
    let arguments = ProcessInfo.processInfo.arguments
    let generationProvider: any GenerationProvider =
      arguments.contains("-H3ddleFastFakeGeneration")
      ? FakeGenerationProvider(stepDelay: .milliseconds(20))
      : FakeGenerationProvider()

    #if DEBUG
      if let marker = arguments.firstIndex(of: "-H3ddleUITestActiveManagedDownload"),
        arguments.indices.contains(marker + 1)
      {
        // Keep the fixture completely separate from a developer's real model
        // library; the test is about button state, not package discovery.
        let root = FileManager.default.temporaryDirectory
          .appendingPathComponent("H3ddleUITestModels-\(ProcessInfo.processInfo.processIdentifier)")
        let model = AppModel(
          generationProvider: generationProvider,
          modelDownloader: ModelPackageDownloader(store: ModelPackageStore(rootURL: root))
        )
        model.prepareManagedDownloadFixture(packageID: arguments[marker + 1])
        return model
      }
    #endif

    return AppModel(generationProvider: generationProvider)
  }()

  var body: some Scene {
    WindowGroup("H3ddle") {
      EditorView(model: model)
        .frame(minWidth: 1_080, minHeight: 700)
        .preferredColorScheme(.dark)
        .background(WindowChromeInstaller())
        .onReceive(
          NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)
        ) { _ in
          DockAttention.dismissFinishedMarker()
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
