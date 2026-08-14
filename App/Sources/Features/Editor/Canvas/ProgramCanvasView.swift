import AVFoundation
import AppKit
import H3ddleCore
import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI

struct ProgramCanvasView: View {
  @Bindable var model: AppModel
  @State private var viewport = CanvasViewport()
  @State private var lastPan: CGSize = .zero
  @State private var pinchBase: CGFloat?
  @State private var pointerInside = false
  @State private var eventMonitor: Any?
  @State private var videoNaturalSizes: [URL: CGSize] = [:]

  var body: some View {
    GeometryReader { proxy in
      ZStack {
        RadialGradient(
          colors: [H3Color.gradientTop, H3Color.canvas],
          center: UnitPoint(x: 0.5, y: 0.38),
          startRadius: 0,
          endRadius: max(proxy.size.width, proxy.size.height)
        )

        monitorSurface
          .padding(32)
          .scaleEffect(viewport.magnification)
          .offset(viewport.offset)
          .frame(width: proxy.size.width, height: proxy.size.height)
          .clipped()
          .allowsHitTesting(false)

        Color.clear
          .contentShape(Rectangle())
          .highPriorityGesture(panGesture)
          .simultaneousGesture(magnifyGesture)
          .simultaneousGesture(resetGesture)
          .onHover { pointerInside = $0 }
          .accessibilityIdentifier("program-preview")
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    .onAppear {
      syncPlayback()
      installPanMonitor()
    }
    .onDisappear(perform: removePanMonitor)
    .onChange(of: model.playback.clock.currentTime) { _, _ in
      syncPlayback()
    }
    .onChange(of: model.project.timeline) { _, _ in
      syncPlayback()
    }
    .onChange(of: model.project.settings) { _, _ in
      syncPlayback()
    }
    .onChange(of: model.visualLaneAudible) { _, _ in
      syncPlayback()
    }
    .onChange(of: model.audioLaneAudible) { _, _ in
      syncPlayback()
    }
  }

  @ViewBuilder
  private var monitorSurface: some View {
    ZStack {
      projectBackground
      mediaSurface
    }
    .aspectRatio(model.project.settings.aspectFraction, contentMode: .fit)
    .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
    .overlay {
      RoundedRectangle(cornerRadius: 4, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
  }

  @ViewBuilder
  private var mediaSurface: some View {
    switch currentFrame.visual {
    case .video(let asset, _, _):
      PlacedCanvasMedia(
        source: videoNaturalSizes[asset.url] ?? CGSize(width: 16, height: 9),
        transform: currentFrame.visualTransform
      ) {
        ProgramPlayerLayer(player: model.playback.visualPlayer)
      }
      .task(id: asset.url) {
        await rememberVideoSize(asset.url)
      }
    case .image(let asset):
      if let image = NSImage(contentsOf: asset.url) {
        PlacedCanvasMedia(source: image.size, transform: currentFrame.visualTransform) {
          Image(nsImage: image)
            .resizable()
        }
      }
    case .empty:
      if model.project.timeline.visualItems.isEmpty {
        emptyState
      }
    }
  }

  private func rememberVideoSize(_ url: URL) async {
    if videoNaturalSizes[url] != nil { return }
    guard let size = await MediaSourceSize.video(at: url) else { return }
    videoNaturalSizes[url] = size
  }

  @ViewBuilder
  private var projectBackground: some View {
    if model.project.settings.background.isClear {
      H3Checkerboard(cell: 10)
    } else {
      Color(h3Hex: model.project.settings.background.rawValue)
    }
  }

  private var emptyState: some View {
    VStack(spacing: 12) {
      Image(systemName: "sparkles")
        .font(.system(size: 24, weight: .medium))
        .foregroundStyle(H3Color.accent)
      Text("Generate the first visual")
        .font(.system(size: 14, weight: .semibold))
      Text("Generated video and audio will be playable here.")
        .font(.system(size: 12))
        .foregroundStyle(H3Color.textSecondary)
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }

  private var panGesture: some Gesture {
    DragGesture(minimumDistance: 2)
      .onChanged { value in
        let delta = CGSize(
          width: value.translation.width - lastPan.width,
          height: value.translation.height - lastPan.height
        )
        lastPan = value.translation
        viewport.pan(by: delta)
      }
      .onEnded { _ in
        lastPan = .zero
      }
  }

  private var magnifyGesture: some Gesture {
    MagnifyGesture()
      .onChanged { value in
        if pinchBase == nil {
          pinchBase = viewport.magnification
        }
        if let pinchBase {
          viewport.magnification = min(
            max(pinchBase * value.magnification, CanvasViewport.minimumMagnification),
            CanvasViewport.maximumMagnification
          )
        }
      }
      .onEnded { _ in
        pinchBase = nil
      }
  }

  private var resetGesture: some Gesture {
    TapGesture(count: 2).onEnded {
      viewport.reset()
    }
  }

  private func installPanMonitor() {
    removePanMonitor()
    eventMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
      guard pointerInside, model.activeGenerationKind == nil,
        !event.modifierFlags.contains(.control)
      else {
        return event
      }
      let scale: CGFloat = event.hasPreciseScrollingDeltas ? 1 : 10
      viewport.pan(
        by: CGSize(
          width: event.scrollingDeltaX * scale,
          height: event.scrollingDeltaY * scale
        )
      )
      return nil
    }
  }

  private func removePanMonitor() {
    if let eventMonitor {
      NSEvent.removeMonitor(eventMonitor)
      self.eventMonitor = nil
    }
  }

  private var currentFrame: ProgramPreviewFrame {
    ProgramPreview.frame(
      at: model.playback.clock.currentTime,
      project: model.project,
      visualMuted: !model.visualLaneAudible,
      audioMuted: !model.audioLaneAudible
    )
  }

  private func syncPlayback() {
    model.playback.sync(
      project: model.project,
      visualMuted: !model.visualLaneAudible,
      audioMuted: !model.audioLaneAudible
    )
  }
}

private enum MediaSourceSize {
  static func video(at url: URL) async -> CGSize? {
    let asset = AVURLAsset(url: url)
    guard let track = try? await asset.loadTracks(withMediaType: .video).first else {
      return nil
    }
    let natural = (try? await track.load(.naturalSize)) ?? .zero
    let transform = (try? await track.load(.preferredTransform)) ?? .identity
    let mapped = natural.applying(transform)
    let width = abs(mapped.width)
    let height = abs(mapped.height)
    guard width > 1, height > 1 else { return nil }
    return CGSize(width: width, height: height)
  }
}
