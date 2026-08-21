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
  @State private var mediaSizes: [URL: CanvasMediaDimensions] = [:]

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
          CanvasGizmoOverlay(layout: gizmo, badge: gizmoBadge)
            .accessibilityIdentifier("canvas-gizmo")
        }

        Color.clear
          .contentShape(Rectangle())
          .simultaneousGesture(magnifyGesture)
          .simultaneousGesture(resetGesture)
          .onHover { pointerInside = $0 }

        CanvasInteractionProbe(onEvent: handlePointer, cursorAt: { cursor(at: $0) })
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
    .onChange(of: model.textLaneAudible) { _, _ in refreshPreview() }
    .onChange(of: model.canvasGesture) { _, gesture in
      if case .text = gesture?.target { return }
      refreshPreview()
    }
    .onChange(of: monitorSize) { _, _ in refreshPreview() }
    .task(id: model.project.assets) {
      await loadMediaSizes(for: model.project.assets)
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
    if !hasComposableContent {
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
    case .rotate(let target):
      startGesture(target, kind: .rotate, at: sample)
    case .scale(let target, let corner):
      startGesture(target, kind: .scale(corner), at: sample)
    case .body(let target):
      select(target)
      startGesture(target, kind: .move, at: sample)
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
    guard var session = model.canvasGesture,
      let target = canvasObject(session.target),
      let program = unboundedProgramPoint(sample.viewPoint)
    else {
      return
    }
    session.shiftDown = sample.shift
    session.commandDown = sample.command
    let source = sourceSize(for: target)
    let usesMediaFit = target.isVisual
    switch session.kind {
    case .move:
      session.current = CanvasGestureMath.moved(
        origin: session.origin,
        deltaProgramX: program.x - session.startProgram.x,
        deltaProgramY: program.y - session.startProgram.y
      )
    case .scale(let corner):
      session.current = CanvasGestureMath.scaled(
        origin: session.origin,
        grab: corner,
        pointer: program,
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        sourceWidth: source.width,
        sourceHeight: source.height,
        aboutCenter: sample.command,
        usesMediaFit: usesMediaFit
      )
    case .rotate:
      session.current = CanvasGestureMath.rotated(
        origin: session.origin,
        start: session.startProgram,
        pointer: program,
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        sourceWidth: source.width,
        sourceHeight: source.height,
        snapToIncrements: sample.shift,
        usesMediaFit: usesMediaFit
      )
    }
    model.canvasGesture = session
    if case .text = session.target {
      blitLiveTextOverlays()
    }
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
    _ target: CanvasObject,
    kind: CanvasGestureSession.Kind,
    at sample: CanvasPointerSample
  ) {
    guard let program = unboundedProgramPoint(sample.viewPoint) else { return }
    model.canvasGesture = CanvasGestureSession(
      target: target.timelineID,
      kind: kind,
      origin: target.transform,
      current: target.transform,
      startProgram: program,
      shiftDown: sample.shift,
      commandDown: sample.command
    )
  }

  private enum CanvasObject: Equatable {
    case visual(VisualItem)
    case text(TextItem)

    var timelineID: TimelineItemID {
      switch self {
      case .visual(let item): .visual(item.id)
      case .text(let item): .text(item.id)
      }
    }

    var transform: CanvasObjectTransform {
      switch self {
      case .visual(let item): item.canvasTransform
      case .text(let item): item.canvasTransform
      }
    }

    var isEnabled: Bool {
      switch self {
      case .visual(let item): item.isEnabled
      case .text(let item): item.isEnabled
      }
    }

    var isVisual: Bool {
      if case .visual = self { return true }
      return false
    }
  }

  private enum Hit: Equatable {
    case rotate(CanvasObject)
    case scale(CanvasObject, CanvasCorner)
    case body(CanvasObject)
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
        if CanvasGizmoGeometry.cornerIntent(at: point, corner: corner, in: layout) == .rotate {
          return .rotate(target)
        }
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
    if let program {
      for item in textsAtPlayhead().reversed() {
        if hitsText(item, program: program, selected: isSelectedText(item.id)) {
          return .body(.text(item))
        }
      }
      if let item = visualAtPlayhead(), item.isEnabled, !model.visualTrackMuted {
        if CanvasGizmoGeometry.contains(
          program: program,
          placement: placement(for: .visual(item)),
          canvas: canvas,
          tolerance: 3
        ) {
          return .body(.visual(item))
        }
      }
    }
    return .empty
  }

  private func cursor(at viewPoint: CGPoint) -> NSCursor? {
    guard let layout = gizmoLayout, gizmoTarget?.isEnabled == true else { return nil }
    let point = (x: Double(viewPoint.x), y: Double(viewPoint.y))
    let center = CanvasGizmoGeometry.centroid(of: layout)
    if CanvasGizmoGeometry.hitsRotate(at: point, in: layout) {
      let radial = atan2(
        layout.rotateHandle.y - center.y,
        layout.rotateHandle.x - center.x
      )
      return CanvasGizmoCursor.rotate(radians: CanvasGizmoCursor.rotateRadians(radial: radial))
    }
    guard
      let corner = CanvasGizmoGeometry.hitCorner(at: point, in: layout),
      let handle = layout.corners[corner]
    else {
      return nil
    }
    let radial = atan2(handle.y - center.y, handle.x - center.x)
    if CanvasGizmoGeometry.cornerIntent(at: point, corner: corner, in: layout) == .rotate {
      return CanvasGizmoCursor.rotate(radians: CanvasGizmoCursor.rotateRadians(radial: radial))
    }
    return CanvasGizmoCursor.scale(radians: radial)
  }

  private func presentClipMenu(at local: CGPoint) {
    let origin = CGPoint(
      x: local.x,
      y: local.y
    )
    if case .body(let target) = hit(at: local) {
      select(target)
      switch target {
      case .visual(let item):
        clipMenu = ClipMenuPlacement(target: .visual(item.id), origin: mappedMenuOrigin(local))
      case .text(let item):
        clipMenu = ClipMenuPlacement(target: .text(item.id), origin: mappedMenuOrigin(local))
      }
      return
    }
    clipMenu = ClipMenuPlacement(target: .insertText, origin: mappedMenuOrigin(origin))
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

  private var gizmoTarget: CanvasObject? {
    switch model.selectedTimelineItem {
    case .visual(let id):
      guard !model.visualTrackMuted,
        let item = model.project.timeline.visualItems.first(where: { $0.id == id }),
        item.isEnabled, visualAtPlayhead()?.id == id
      else {
        return nil
      }
      return .visual(item)
    case .text(let id):
      guard !model.textTrackMuted,
        let item = model.project.timeline.textItems.first(where: { $0.id == id }),
        item.isEnabled, textsAtPlayhead().contains(where: { $0.id == id })
      else {
        return nil
      }
      return .text(item)
    case .audio, nil:
      return nil
    }
  }

  private var gizmoBadge: String? {
    guard let gesture = model.canvasGesture else { return nil }
    switch gesture.kind {
    case .rotate:
      return "\(Self.displayDegrees(gesture.current.rotationRadians))°"
    case .scale:
      return "\(Int((gesture.current.scale * 100).rounded()))%"
    case .move:
      return nil
    }
  }

  private static func displayDegrees(_ radians: Double) -> Int {
    var degrees = radians * 180 / .pi
    degrees = degrees.truncatingRemainder(dividingBy: 360)
    if degrees > 180 { degrees -= 360 }
    if degrees <= -180 { degrees += 360 }
    return Int(degrees.rounded())
  }

  private var gizmoLayout: CanvasGizmoGeometry.Layout? {
    guard let target = gizmoTarget, viewSize.width > 1 else { return nil }
    return CanvasGizmoGeometry.layout(
      placement: placement(for: target),
      canvas: canvas,
      viewSize: (Double(viewSize.width), Double(viewSize.height)),
      aspect: Double(model.project.settings.aspectFraction),
      magnification: Double(viewport.magnification),
      offset: (Double(viewport.offset.width), Double(viewport.offset.height))
    )
  }

  private func placement(for target: CanvasObject) -> CanvasLayout.Placement {
    let transform = liveTransform(for: target)
    let source = sourceSize(for: target)
    switch target {
    case .visual:
      return CanvasLayout.placed(
        sourceWidth: source.width,
        sourceHeight: source.height,
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        transform: transform
      )
    case .text:
      return CanvasLayout.overlayPlaced(
        sourceWidth: source.width,
        sourceHeight: source.height,
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        transform: transform
      )
    }
  }

  private func sourceSize(for target: CanvasObject) -> (width: Double, height: Double) {
    switch target {
    case .visual(let item):
      if let asset = model.project.asset(id: item.assetID) {
        let size = mediaSizes[asset.url] ?? .fallback
        return (size.width, size.height)
      }
      return (canvas.width, canvas.height)
    case .text(let item):
      let layout = TextRasterizer.layout(
        item,
        layoutSize: (model.project.settings.width, model.project.settings.height)
      )
      return layout.expandedSize
    }
  }

  private func liveTransform(for target: CanvasObject) -> CanvasObjectTransform {
    if let gesture = model.canvasGesture, gesture.target == target.timelineID {
      return gesture.current
    }
    return target.transform
  }

  private func select(_ target: CanvasObject) {
    model.selectedTimelineItem = target.timelineID
  }

  private func canvasObject(_ id: TimelineItemID) -> CanvasObject? {
    switch id {
    case .visual(let uuid):
      return model.project.timeline.visualItems.first { $0.id == uuid }.map(CanvasObject.visual)
    case .text(let uuid):
      return model.project.timeline.textItems.first { $0.id == uuid }.map(CanvasObject.text)
    case .audio:
      return nil
    }
  }

  private func textsAtPlayhead() -> [TextItem] {
    guard !model.textTrackMuted else { return [] }
    let time = model.playback.clock.currentTime
    return model.project.timeline.textItems.filter { item in
      item.isEnabled && time >= item.startTime && time < item.endTime
    }
  }

  private func isSelectedText(_ id: UUID) -> Bool {
    model.selectedTimelineItem == .text(id)
  }

  private func hitsText(
    _ item: TextItem,
    program: (x: Double, y: Double),
    selected: Bool
  ) -> Bool {
    let layout = TextRasterizer.layout(
      item,
      layoutSize: (model.project.settings.width, model.project.settings.height)
    )
    let placement = placement(for: .text(item))
    if selected {
      return CanvasGizmoGeometry.contains(
        program: program,
        placement: placement,
        canvas: canvas,
        tolerance: 3
      )
    }
    let point = (program.x * canvas.width, program.y * canvas.height)
    for glyph in layout.glyphBounds {
      let rect = CanvasLayout.Rect(
        x: glyph.x + layout.contentInset,
        y: glyph.y + layout.contentInset,
        width: glyph.width,
        height: glyph.height
      )
      let quad = CanvasLayout.overlayQuad(
        rect: rect,
        sourceWidth: layout.expandedSize.width,
        sourceHeight: layout.expandedSize.height,
        placement: placement
      )
      if CanvasGestureMath.contains(point: point, quad: quad, tolerance: 2) {
        return true
      }
    }
    return false
  }

  private var hasComposableContent: Bool {
    !model.project.timeline.visualItems.isEmpty
      || !model.project.timeline.textItems.isEmpty
  }

  private func loadMediaSizes(for assets: [AssetReference]) async {
    let visualAssets = assets.filter { $0.kind != .audio }
    let resolved = await withTaskGroup(
      of: (URL, CanvasMediaDimensions).self,
      returning: [URL: CanvasMediaDimensions].self
    ) { group in
      for asset in visualAssets {
        group.addTask {
          (asset.url, await CanvasMediaSize.shared.size(for: asset))
        }
      }
      var sizes: [URL: CanvasMediaDimensions] = [:]
      for await (url, size) in group {
        sizes[url] = size
      }
      return sizes
    }
    guard !Task.isCancelled else { return }
    mediaSizes = resolved
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

  private func unboundedProgramPoint(_ viewPoint: CGPoint) -> (x: Double, y: Double)? {
    CanvasViewportMath.unboundedProgramPoint(
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

  private func blitLiveTextOverlays() {
    var overrides: [UUID: CanvasObjectTransform] = [:]
    if let gesture = model.canvasGesture {
      switch gesture.target {
      case .visual(let id), .text(let id):
        overrides[id] = gesture.current
      case .audio:
        break
      }
    }
    let frame = model.playback.sync(
      project: model.project,
      visualMuted: !model.visualLaneAudible,
      audioMuted: !model.audioLaneAudible,
      textMuted: !model.textLaneAudible,
      transformOverrides: overrides
    )
    if !presenter.blitOverlays(frame.overlays) {
      refreshPreview()
    }
  }

  private func refreshPreview() {
    var overrides: [UUID: CanvasObjectTransform] = [:]
    if let gesture = model.canvasGesture {
      switch gesture.target {
      case .visual(let id), .text(let id):
        overrides[id] = gesture.current
      case .audio:
        break
      }
    }
    let frame = model.playback.sync(
      project: model.project,
      visualMuted: !model.visualLaneAudible,
      audioMuted: !model.audioLaneAudible,
      textMuted: !model.textLaneAudible,
      transformOverrides: overrides
    )
    guard hasComposableContent else {
      presenter.clear()
      return
    }
    let videoFrame: CGImage?
    if case .video(_, let localTime, _) = frame.visual {
      if model.playback.isPlaying {
        guard let liveFrame = model.playback.copyCurrentVisualVideoImage() else { return }
        videoFrame = liveFrame
      } else {
        videoFrame = model.playback.copyVisualVideoImage(matching: localTime)
      }
    } else {
      videoFrame = nil
    }
    let renderScale: CGFloat = model.canvasGesture == nil
      ? NSScreen.main?.backingScaleFactor ?? 2
      : 1
    presenter.render(
      frame: frame,
      canvas: canvasSize,
      scale: renderScale,
      background: model.project.settings.background,
      videoFrame: videoFrame,
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
