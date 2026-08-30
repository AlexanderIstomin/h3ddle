import AppKit
import H3ddleCore
import UniformTypeIdentifiers

enum AssetFileExporter {
  @MainActor
  static func presentDownload(of asset: AssetReference) {
    let panel = NSSavePanel()
    panel.title = "Download"
    panel.prompt = "Save"
    panel.canCreateDirectories = true
    panel.isExtensionHidden = false

    let fileExtension = asset.url.pathExtension.isEmpty
      ? defaultExtension(for: asset.kind)
      : asset.url.pathExtension
    panel.allowedContentTypes = [contentType(for: fileExtension, kind: asset.kind)]
    panel.nameFieldStringValue =
      asset.displayName.replacingOccurrences(of: "/", with: "-")
      + (asset.displayName.lowercased().hasSuffix(".\(fileExtension.lowercased())")
        ? ""
        : ".\(fileExtension)")

    guard panel.runModal() == .OK, let destination = panel.url else { return }
    guard destination.standardizedFileURL != asset.url.standardizedFileURL else { return }

    let fileManager = FileManager.default
    do {
      if fileManager.fileExists(atPath: destination.path) {
        try fileManager.removeItem(at: destination)
      }
      try fileManager.copyItem(at: asset.url, to: destination)
    } catch {
      let alert = NSAlert()
      alert.messageText = "Couldn’t download file"
      alert.informativeText = error.localizedDescription
      alert.alertStyle = .warning
      alert.runModal()
    }
  }

  private static func defaultExtension(for kind: MediaKind) -> String {
    switch kind {
    case .video: "mp4"
    case .image: "png"
    case .audio: "wav"
    }
  }

  private static func contentType(for fileExtension: String, kind: MediaKind) -> UTType {
    if let type = UTType(filenameExtension: fileExtension) { return type }
    switch kind {
    case .video: return .movie
    case .image: return .image
    case .audio: return .audio
    }
  }
}
