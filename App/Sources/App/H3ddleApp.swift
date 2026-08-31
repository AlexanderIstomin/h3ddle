import AppKit
import H3ddleGeneration
import H3ddleModels
import SwiftUI

@main
struct H3ddleApp: App {
  @State private var model: AppModel = {
    let arguments = ProcessInfo.processInfo.arguments
    let activeQueueFixture = arguments.contains("-H3ddleUITestActiveQueueJob")
    let generationProvider: any GenerationProvider
    if activeQueueFixture {
      generationProvider = FakeGenerationProvider(stepDelay: .seconds(30))
    } else if arguments.contains("-H3ddleFastFakeGeneration") {
      generationProvider = FakeGenerationProvider(stepDelay: .milliseconds(20))
    } else {
      generationProvider = MissingModelGenerationProvider()
    }
    let testQueueStore = GenerationQueueStore(
      rootURL: FileManager.default.temporaryDirectory.appendingPathComponent(
        "H3ddleUITestQueue-\(ProcessInfo.processInfo.processIdentifier)",
        isDirectory: true
      )
    )

    #if DEBUG
      if activeQueueFixture {
        let model = AppModel(
          generationProvider: generationProvider,
          generationQueueStore: testQueueStore
        )
        model.prepareActiveGenerationQueueFixture()
        return model
      }
      if let marker = arguments.firstIndex(of: "-H3ddleUITestActiveManagedDownload"),
        arguments.indices.contains(marker + 1)
      {
        // Keep the fixture completely separate from a developer's real model
        // library; the test is about button state, not package discovery.
        let root = FileManager.default.temporaryDirectory
          .appendingPathComponent("H3ddleUITestModels-\(ProcessInfo.processInfo.processIdentifier)")
        let model = AppModel(
          generationProvider: generationProvider,
          modelDownloader: ModelPackageDownloader(store: ModelPackageStore(rootURL: root)),
          generationQueueStore: testQueueStore
        )
        model.prepareManagedDownloadFixture(packageID: arguments[marker + 1])
        return model
      }
    #endif

    return AppModel(
      generationProvider: generationProvider,
      generationQueueStore: arguments.contains("-H3ddleFastFakeGeneration")
        ? testQueueStore : GenerationQueueStore()
    )
  }()

  var body: some Scene {
    WindowGroup("H3ddle") {
      EditorView(model: model)
        .frame(minWidth: 1_080, minHeight: 700)
        .preferredColorScheme(.dark)
        .background(WindowChromeInstaller())
        .task {
          model.resumeInterruptedGenerationIfAvailable()
        }
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
      CommandGroup(replacing: .newItem) {
        Button("New Project") {
          model.createNewProject()
        }
        .keyboardShortcut("n")
      }
    }
  }
}
