import AppKit
import AVFoundation
import AVKit
import H3ddleCore
import H3ddleDesignSystem
import SwiftUI
import UniformTypeIdentifiers

struct GeneratedAssetPreview: View {
  let asset: AssetReference
  let generationDuration: String?

  @State private var saveNotice: SaveNotice?

  var body: some View {
    ZStack(alignment: .top) {
      media

      HStack(spacing: H3Spacing.small) {
        VStack(alignment: .leading, spacing: 2) {
          Text(asset.displayName)
            .font(.system(size: 12, weight: .semibold))
          HStack(spacing: H3Spacing.small) {
            Text(String(format: "%.2f s", asset.duration))
            if let generationDuration {
              Text("Generated in \(generationDuration)")
            }
          }
          .font(.system(size: 10, weight: .medium, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
        }

        Spacer()

        Button("Save Copy…") {
          saveCopy()
        }
        .buttonStyle(H3QuietButtonStyle())
        .accessibilityIdentifier("save-generated-asset")
      }
      .padding(.horizontal, H3Spacing.medium)
      .padding(.vertical, H3Spacing.small)
      .background(.ultraThinMaterial)
    }
    .background(Color.black)
    .alert(item: $saveNotice) { notice in
      Alert(
        title: Text(notice.title),
        message: Text(notice.message),
        dismissButton: .default(Text("OK"))
      )
    }
  }

  @ViewBuilder
  private var media: some View {
    switch asset.kind {
    case .video:
      NativeVideoPlayer(url: asset.url)
        .accessibilityIdentifier("generated-video-player")
    case .image:
      LocalImagePreview(url: asset.url) {
        unavailableMedia
      }
    case .audio:
      AudioPreviewPlayer(url: asset.url, duration: asset.duration)
    }
  }

  private var unavailableMedia: some View {
    ContentUnavailableView(
      "Preview unavailable",
      systemImage: "exclamationmark.triangle",
      description: Text(asset.url.lastPathComponent)
    )
    .foregroundStyle(H3Color.textSecondary)
  }

  private func saveCopy() {
    do {
      guard let savedURL = try GeneratedAssetSaver.saveCopy(of: asset) else { return }
      saveNotice = SaveNotice(
        title: "Video saved",
        message: savedURL.path(percentEncoded: false)
      )
    } catch {
      saveNotice = SaveNotice(
        title: "Couldn’t save video",
        message: error.localizedDescription
      )
    }
  }
}

struct NativeVideoPlayer: NSViewRepresentable {
  let url: URL

  func makeNSView(context: Context) -> AVPlayerView {
    let view = AVPlayerView()
    view.controlsStyle = .floating
    view.videoGravity = .resizeAspect
    view.player = AVPlayer(url: url)
    return view
  }

  func updateNSView(_ view: AVPlayerView, context: Context) {
    let currentURL = (view.player?.currentItem?.asset as? AVURLAsset)?.url
    guard currentURL != url else { return }
    view.player?.pause()
    view.player = AVPlayer(url: url)
  }

  static func dismantleNSView(_ view: AVPlayerView, coordinator: ()) {
    view.player?.pause()
    view.player = nil
  }
}

@MainActor
private enum GeneratedAssetSaver {
  static func saveCopy(of asset: AssetReference) throws -> URL? {
    let panel = NSSavePanel()
    panel.title = "Save Generated Video"
    panel.prompt = "Save Copy"
    panel.canCreateDirectories = true
    panel.isExtensionHidden = false

    let fileExtension = asset.url.pathExtension.isEmpty ? "mp4" : asset.url.pathExtension
    panel.allowedContentTypes = [contentType(for: fileExtension)]
    panel.nameFieldStringValue =
      asset.displayName.replacingOccurrences(of: "/", with: "-") + "." + fileExtension

    guard panel.runModal() == .OK, let destination = panel.url else { return nil }
    guard destination.standardizedFileURL != asset.url.standardizedFileURL else {
      return destination
    }

    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: destination.path) {
      try fileManager.removeItem(at: destination)
    }
    try fileManager.copyItem(at: asset.url, to: destination)
    return destination
  }

  private static func contentType(for fileExtension: String) -> UTType {
    UTType(filenameExtension: fileExtension) ?? .movie
  }
}

private struct SaveNotice: Identifiable {
  let id = UUID()
  let title: String
  let message: String
}
