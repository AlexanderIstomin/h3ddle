import AppKit
import H3ddleCore
import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI

struct ProgramCanvasView: View {
  @Bindable var model: AppModel
  @Binding var clipMenu: ClipMenuPlacement?
  @State private var viewport = CanvasViewport()
  @State private var pinchBase: CGFloat?
  @State private var pointerInside = false
  @State private var eventMonitor: Any?
  @State private var presenter = ProgramFramePresenter()
  @State private var monitorSize: CGSize = .zero
  @State private var viewSize: CGSize = .zero
  @State private var pointerDown: CanvasPointerSample?
  @State private var lastViewPoint: CGPoint = .zero
  @State private var isPanning = false
  @State private var panFromEmpty = false
  @State private var didDrag = false
  @State private var canvasFrame: CGRect = .zero

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

        if let gizmo = gizmoLayout {
          CanvasGizmoOverlay(layout: gizmo)
            .accessibilityIdentifier("canvas-gizmo")
        }

        Color.clear
          .contentShape(Rectangle())
          .simultaneousGesture(magnifyGesture)
          .simultaneousGesture(resetGesture)
          .onHover { pointerInside = $0 }

        CanvasInteractionProbe(onEvent: handlePointer)
          .allowsHitTesting(true)
      }
      .coordinateSpace(name: "program-canvas")
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .background {
        GeometryReader { frameProxy in
          Color.clear.preference(
            key: CanvasFrameKey.self,
            value: frameProxy.frame(in: .named("editor-root"))
          )
        }
      }
      .onPreferenceChange(CanvasFrameKey.self) { canvasFrame = $0 }
      .onAppear { viewSize = proxy.size }
      .onChange(of: proxy.size) { _, size in viewSize = size }
    }
    .accessibilityIdentifier("program-preview")
    .onAppear {
      refreshPreview()
      installPanMonitor()
    }
    .onDisappear(perform: removePanMonitor)
    .onChange(of: model.playback.clock.currentTime) { _, _ in refreshPreview() }
    .onChange(of: model.project.timeline) { _, _ in refreshPreview() }
    .onChange(of: model.project.settings) { _, _ in refreshPreview() }
    .onChange(of: model.visualLaneAudible) { _, _ in refreshPreview() }
    .onChange(of: model.audioLaneAudible) { _, _ in refreshPreview() }
    .onChange(of: model.canvasGesture) { _, _ in refreshPreview() }
    .onChange(of: monitorSize) { _, _ in refreshPreview() }
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

  private func handlePointer(_ event: CanvasPointerEvent) {
    switch event {
    case .doubleClick:
      viewport.reset()
      model.canvasGesture = nil
    case .rightClick(let local):
      presentClipMenu(at: local)
    case .down(let sample):
      beginPointer(sample)
    case .dragged(let sample):
      dragPointer(sample)
    case .up(let sample):
      endPointer(sample)
    }
  }

  private func beginPointer(_ sample: CanvasPointerSample) {
    pointerDown = sample
    lastViewPoint = sample.viewPoint
    didDrag = false
    isPanning = false
    panFromEmpty = false
    if sample.option {
      isPanning = true
      return
    }
    let hit = hit(at: sample.viewPoint)
    switch hit {
    case .rotate(let item):
      startGesture(item, kind: .rotate, at: sample)
    case .scale(let item, let corner):
      startGesture(item, kind: .scale(corner), at: sample)
    case .body(let item):
      model.selectedTimelineItem = .visual(item.id)
      startGesture(item, kind: .move, at: sample)
    case .empty:
      isPanning = true
      panFromEmpty = true
    }
  }

  private func dragPointer(_ sample: CanvasPointerSample) {
    let delta = CGSize(
      width: sample.viewPoint.x - lastViewPoint.x,
      height: sample.viewPoint.y - lastViewPoint.y
    )
    if hypot(delta.width, delta.height) >= 2 { didDrag = true }
    lastViewPoint = sample.viewPoint
    if isPanning {
      viewport.pan(by: delta)
      return
    }
    guard var session = model.canvasGesture, case .visual(let id) = session.target,
      let item = model.project.timeline.visualItems.first(where: { $0.id == id }),
      let program = programPoint(sample.viewPoint)
    else {
      return
    }
    session.shiftDown = sample.shift
    session.commandDown = sample.command
    switch session.kind {
    case .move:
      session.current = CanvasGestureMath.moved(
        origin: session.origin,
        deltaProgramX: program.x - session.startProgram.x,
        deltaProgramY: program.y - session.startProgram.y
      )
    case .scale(let corner):
      let source = sourceSize(for: item)
      session.current = CanvasGestureMath.scaled(
        origin: session.origin,
        grab: corner,
        pointer: program,
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        sourceWidth: source.width,
        sourceHeight: source.height,
        aboutCenter: sample.command
      )
    case .rotate:
      let source = sourceSize(for: item)
      session.current = CanvasGestureMath.rotated(
        origin: session.origin,
        start: session.startProgram,
        pointer: program,
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        sourceWidth: source.width,
        sourceHeight: source.height,
        snapToIncrements: sample.shift
      )
    }
    model.canvasGesture = session
  }

  private func endPointer(_ sample: CanvasPointerSample) {
    defer {
      pointerDown = nil
      isPanning = false
      didDrag = false
    }
    if isPanning {
      if !didDrag, panFromEmpty {
        model.selectedTimelineItem = nil
      }
      return
    }
    if model.canvasGesture != nil {
      if didDrag {
        model.commitCanvasGesture()
      } else {
        model.canvasGesture = nil
      }
    }
  }

  private func startGesture(
    _ item: VisualItem,
    kind: CanvasGestureSession.Kind,
    at sample: CanvasPointerSample
  ) {
    guard let program = programPoint(sample.viewPoint) else { return }
    model.canvasGesture = CanvasGestureSession(
      target: .visual(item.id),
      kind: kind,
      origin: item.canvasTransform,
      current: item.canvasTransform,
      startProgram: program,
      shiftDown: sample.shift,
      commandDown: sample.command
    )
  }

  private enum Hit: Equatable {
    case rotate(VisualItem)
    case scale(VisualItem, CanvasCorner)
    case body(VisualItem)
    case empty
  }

  private func hit(at viewPoint: CGPoint) -> Hit {
    let program = programPoint(viewPoint)
    if let target = gizmoTarget, let layout = gizmoLayout, target.isEnabled {
      let point = (x: Double(viewPoint.x), y: Double(viewPoint.y))
      if CanvasGizmoGeometry.hitsRotate(at: point, in: layout) {
        return .rotate(target)
      }
      if let corner = CanvasGizmoGeometry.hitCorner(at: point, in: layout) {
        return .scale(target, corner)
      }
      if let program,
        CanvasGizmoGeometry.contains(
          program: program,
          placement: placement(for: target),
          canvas: canvas,
          tolerance: 3
        )
      {
        return .body(target)
      }
    }
    if let program, let item = visualAtPlayhead(), item.isEnabled, !model.visualTrackMuted {
      if CanvasGizmoGeometry.contains(
        program: program,
        placement: placement(for: item),
        canvas: canvas,
        tolerance: 3
      ) {
        return .body(item)
      }
    }
    return .empty
  }

  private func presentClipMenu(at local: CGPoint) {
    let origin = CGPoint(
      x: local.x,
      y: local.y
    )
    if case .body(let item) = hit(at: local) {
      model.selectedTimelineItem = .visual(item.id)
      clipMenu = ClipMenuPlacement(target: .visual(item.id), origin: mappedMenuOrigin(local))
      return
    }
    guard let item = visualAtPlayhead() else { return }
    model.selectedTimelineItem = .visual(item.id)
    clipMenu = ClipMenuPlacement(target: .visual(item.id), origin: mappedMenuOrigin(origin))
  }

  private func mappedMenuOrigin(_ local: CGPoint) -> CGPoint {
    CGPoint(x: canvasFrame.minX + local.x, y: canvasFrame.minY + local.y)
  }

  private func visualAtPlayhead() -> VisualItem? {
    let time = model.playback.clock.currentTime
    return model.project.timeline.visualPlacements.last(where: { placement in
      time >= placement.startTime && time < placement.startTime + placement.item.duration
    })?.item
  }

  private var gizmoTarget: VisualItem? {
    guard !model.visualTrackMuted, case .visual(let id) = model.selectedTimelineItem,
      let item = model.project.timeline.visualItems.first(where: { $0.id == id }),
      item.isEnabled, visualAtPlayhead()?.id == id
    else {
      return nil
    }
    return item
  }

  private var gizmoLayout: CanvasGizmoGeometry.Layout? {
    guard let item = gizmoTarget, viewSize.width > 1 else { return nil }
    return CanvasGizmoGeometry.layout(
      placement: placement(for: item),
      canvas: canvas,
      viewSize: (Double(viewSize.width), Double(viewSize.height)),
      aspect: Double(model.project.settings.aspectFraction),
      magnification: Double(viewport.magnification),
      offset: (Double(viewport.offset.width), Double(viewport.offset.height))
    )
  }

  private func placement(for item: VisualItem) -> CanvasLayout.Placement {
    let override: CanvasObjectTransform?
    if let gesture = model.canvasGesture, case .visual(let id) = gesture.target, id == item.id {
      override = gesture.current
    } else {
      override = nil
    }
    return CanvasGizmoGeometry.placement(
      item: item,
      source: sourceSize(for: item),
      canvas: canvas,
      override: override
    )
  }

  private func sourceSize(for item: VisualItem) -> (width: Double, height: Double) {
    if let asset = model.project.asset(id: item.assetID) {
      return CanvasMediaSize.size(for: asset)
    }
    return (canvas.width, canvas.height)
  }

  private var canvas: (width: Double, height: Double) {
    (Double(model.project.settings.width), Double(model.project.settings.height))
  }

  private func programPoint(_ viewPoint: CGPoint) -> (x: Double, y: Double)? {
    CanvasViewportMath.programPoint(
      viewPoint: (Double(viewPoint.x), Double(viewPoint.y)),
      viewSize: (Double(viewSize.width), Double(viewSize.height)),
      aspect: Double(model.project.settings.aspectFraction),
      padding: CanvasGizmoGeometry.padding,
      magnification: Double(viewport.magnification),
      offset: (Double(viewport.offset.width), Double(viewport.offset.height))
    )
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
    var overrides: [UUID: CanvasObjectTransform] = [:]
    if let gesture = model.canvasGesture, case .visual(let id) = gesture.target {
      overrides[id] = gesture.current
    }
    let frame = ProgramPreview.frame(
      at: model.playback.clock.currentTime,
      project: model.project,
      visualMuted: !model.visualLaneAudible,
      audioMuted: !model.audioLaneAudible,
      transformOverrides: overrides
    )
    presenter.render(
      frame: frame,
      canvas: canvasSize,
      scale: NSScreen.main?.backingScaleFactor ?? 2,
      background: model.project.settings.background,
      videoFrame: nil,
      layoutWidth: model.project.settings.width,
      layoutHeight: model.project.settings.height
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

private struct CanvasFrameKey: PreferenceKey {
  static let defaultValue: CGRect = .zero

  static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
    value = nextValue()
  }
}
