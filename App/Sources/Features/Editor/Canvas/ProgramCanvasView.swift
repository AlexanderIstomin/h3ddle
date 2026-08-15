import AppKit
import H3ddleCore
import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI

struct ProgramCanvasView: View {
  @Bindable var model: AppModel
  @Binding var clipMenu: ClipMenuPlacement?
  @State private var viewport = CanvasViewport()
  @State private var lastPan: CGSize = .zero
  @State private var pinchBase: CGFloat?
  @State private var pointerInside = false
  @State private var eventMonitor: Any?
  @State private var presenter = ProgramFramePresenter()
  @State private var monitorSize: CGSize = .zero

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
          .overlay {
            GeometryReader { proxy in
              SecondaryClickProbe { local in
                presentClipMenu(
                  at: CGPoint(
                    x: proxy.frame(in: .named("editor-root")).minX + local.x,
                    y: proxy.frame(in: .named("editor-root")).minY + local.y
                  )
                )
              }
            }
          }
          .accessibilityIdentifier("program-preview")
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    .onAppear {
      refreshPreview()
      installPanMonitor()
    }
    .onDisappear(perform: removePanMonitor)
    .onChange(of: model.playback.clock.currentTime) { _, _ in
      refreshPreview()
    }
    .onChange(of: model.project.timeline) { _, _ in
      refreshPreview()
    }
    .onChange(of: model.project.settings) { _, _ in
      refreshPreview()
    }
    .onChange(of: model.visualLaneAudible) { _, _ in
      refreshPreview()
    }
    .onChange(of: model.audioLaneAudible) { _, _ in
      refreshPreview()
    }
    .onChange(of: monitorSize) { _, _ in
      refreshPreview()
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
    .background {
      GeometryReader { proxy in
        Color.clear.preference(key: MonitorSizeKey.self, value: proxy.size)
      }
    }
    .onPreferenceChange(MonitorSizeKey.self) { monitorSize = $0 }
  }

  @ViewBuilder
  private var mediaSurface: some View {
    if model.project.timeline.visualItems.isEmpty {
      emptyState
    } else if let image = presenter.image {
      Image(nsImage: image)
        .resizable()
        .interpolation(.high)
        .scaledToFit()
    }
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

  private func presentClipMenu(at origin: CGPoint) {
    let time = model.playback.clock.currentTime
    guard
      let id = model.project.timeline.visualPlacements.last(where: { placement in
        time >= placement.startTime
          && time < placement.startTime + placement.item.duration
      })?.item.id
    else {
      return
    }
    model.selectedTimelineItem = .visual(id)
    clipMenu = ClipMenuPlacement(target: .visual(id), origin: origin)
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

  private func refreshPreview() {
    model.playback.sync(
      project: model.project,
      visualMuted: !model.visualLaneAudible,
      audioMuted: !model.audioLaneAudible
    )
    guard !model.project.timeline.visualItems.isEmpty else {
      presenter.clear()
      return
    }
    let frame = ProgramPreview.frame(
      at: model.playback.clock.currentTime,
      project: model.project,
      visualMuted: !model.visualLaneAudible,
      audioMuted: !model.audioLaneAudible
    )
    presenter.render(
      frame: frame,
      canvas: canvasSize,
      scale: NSScreen.main?.backingScaleFactor ?? 2,
      background: model.project.settings.background,
      videoFrame: nil
    )
  }

  private var canvasSize: CGSize {
    if monitorSize.width > 2, monitorSize.height > 2 { return monitorSize }
    let aspect = max(model.project.settings.aspectFraction, 0.3)
    return CGSize(width: 1_280, height: 1_280 / aspect)
  }
}

private struct MonitorSizeKey: PreferenceKey {
  static let defaultValue: CGSize = .zero

  static func reduce(value: inout CGSize, nextValue: () -> CGSize) {
    value = nextValue()
  }
}
