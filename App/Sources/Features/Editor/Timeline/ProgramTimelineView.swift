import AppKit
import H3ddleCore
import H3ddleDesignSystem
import H3ddleGeneration
import H3ddleMedia
import SwiftUI

struct ProgramTimelineView: View {
  @Bindable var model: AppModel
  @Binding var appendMenu: AppendMenuPlacement?
  @Binding var clipMenu: ClipMenuPlacement?
  @State private var playheadDragOrigin: TimeInterval?
  @State private var appendButtonFrames: [String: CGRect] = [:]
  @State private var timelineScrollPosition = ScrollPosition(edge: .leading)
  @State private var timelineScrollX: CGFloat = 0
  @State private var timelineViewportWidth: CGFloat = 1
  @State private var pinchBaseZoom: Double?
  @State private var pinchAccumulatedScale = 1.0
  @State private var eventMonitor: Any?
  @State private var pointerInLanes: CGFloat?
  @State private var visualTrim: VisualTrimSession?
  @State private var audioTrim: AudioTrimSession?
  @State private var visualMove: ClipMoveSession?
  @State private var audioMove: ClipMoveSession?
  @State private var transitionDrag: TransitionDragSession?

  private let editorSpace = "editor-root"
  private let pinchResponse = 3.0
  private let wheelSensitivity = 0.004

  private var metrics: TimelineMetrics {
    TimelineMetrics(zoom: model.timelineZoom)
  }

  private var contentDuration: TimeInterval {
    TimelineRuler.contentDuration(
      visualDuration: model.project.timeline.visualDuration,
      audioTrackEnd: model.project.timeline.audioTrackEnd
    )
  }

  private var contentWidth: CGFloat {
    metrics.x(for: contentDuration) + CGFloat(TimelineRuler.trailingLabelPadding)
  }

  var body: some View {
    VStack(spacing: 0) {
      if model.timelineMode == .collapsed {
        collapsedScrubber
      } else {
        expandedTimeline
      }
    }
    .background(H3Color.surface)
  }

  private var expandedTimeline: some View {
    HStack(alignment: .top, spacing: 0) {
      TrackHeaderColumn(model: model)
      ScrollView(.horizontal) {
        ZStack(alignment: .topLeading) {
          VStack(spacing: 0) {
            TimeRulerView(
              duration: contentDuration,
              metrics: metrics,
              visibleX: timelineScrollX,
              visibleWidth: timelineViewportWidth,
              onSeek: model.seekPlayback
            )
              .frame(height: TimelineChrome.rulerHeight)
            if model.showsEffectLanes {
              effectLane(items: model.fxLanesExpanded ? [] : model.effectLaneItems)
              if model.fxLanesExpanded {
                ForEach(model.effectLaneItems) { item in
                  effectLane(items: [item])
                }
              }
            }
            visualLane
            audioLane
          }
          .frame(width: contentWidth, alignment: .topLeading)

          playhead
        }
      }
      .scrollIndicators(.hidden)
      .scrollPosition($timelineScrollPosition)
      .onScrollGeometryChange(for: CGFloat.self) { geometry in
        geometry.contentOffset.x
      } action: { _, offset in
        timelineScrollX = offset
      }
      .onScrollGeometryChange(for: CGFloat.self) { geometry in
        geometry.containerSize.width
      } action: { _, width in
        timelineViewportWidth = max(width, 1)
      }
      .onContinuousHover { phase in
        switch phase {
        case .active(let point):
          pointerInLanes = point.x
        case .ended:
          pointerInLanes = nil
        }
      }
    }
    .frame(
      height: TimelineChrome.bodyHeight(
        showsEffectLanes: model.showsEffectLanes,
        expandedEffectCount: model.fxLanesExpanded ? model.effectLaneItems.count : 0
      )
    )
    .onAppear(perform: installZoomMonitor)
    .onDisappear(perform: removeZoomMonitor)
  }

  private func installZoomMonitor() {
    removeZoomMonitor()
    eventMonitor = NSEvent.addLocalMonitorForEvents(matching: [.magnify, .scrollWheel]) { event in
      guard pointerInLanes != nil else { return event }
      if event.type == .magnify {
        handleMagnify(event)
        return nil
      }
      if event.type == .scrollWheel, event.modifierFlags.contains(.control) {
        handlePinchWheel(event)
        return nil
      }
      return event
    }
  }

  private func removeZoomMonitor() {
    if let eventMonitor {
      NSEvent.removeMonitor(eventMonitor)
      self.eventMonitor = nil
    }
    pinchBaseZoom = nil
    pinchAccumulatedScale = 1
  }

  private func handleMagnify(_ event: NSEvent) {
    if event.phase == .began || pinchBaseZoom == nil {
      pinchBaseZoom = model.timelineZoom
      pinchAccumulatedScale = 1
    }
    pinchAccumulatedScale *= max(0.01, 1 + event.magnification)
    let base = pinchBaseZoom ?? model.timelineZoom
    applyAnchoredZoom(base * pow(pinchAccumulatedScale, pinchResponse))
    if event.phase == .ended || event.phase == .cancelled {
      pinchBaseZoom = nil
      pinchAccumulatedScale = 1
    }
  }

  private func handlePinchWheel(_ event: NSEvent) {
    pinchBaseZoom = nil
    let next = model.timelineZoom * exp(-event.scrollingDeltaY * wheelSensitivity)
    applyAnchoredZoom(next)
  }

  private func applyAnchoredZoom(_ nextZoom: Double) {
    let current = model.timelineZoom
    let zoom = TimelineRuler.clampZoom(nextZoom)
    guard abs(zoom - current) > 0.000_1 else { return }
    let anchor = min(max(pointerInLanes ?? timelineViewportWidth / 2, 0), timelineViewportWidth)
    let nextScroll = TimelineRuler.anchoredScrollOffset(
      currentZoom: current,
      nextZoom: zoom,
      scrollOffset: Double(timelineScrollX),
      anchor: Double(anchor)
    )
    model.setTimelineZoom(zoom)
    timelineScrollPosition.scrollTo(x: nextScroll)
  }

  private var visualLane: some View {
    let renderedPlacements = visualPlacements.filter { placement in
      visualMove?.itemID == placement.item.id
        || shouldRenderTimelineSpan(start: placement.startTime, duration: placement.item.duration)
    }
    return ZStack(alignment: .topLeading) {
      laneBackground(alt: false)
      ForEach(renderedPlacements, id: \.item.id) { placement in
        let asset = model.project.asset(id: placement.item.assetID)
        let isMoving = visualMove?.itemID == placement.item.id
        Group {
        TimelineClipView(
          title: asset?.displayName ?? "Visual",
          kind: asset?.kind ?? .video,
          mediaURL: asset?.url,
          sourceOffset: placement.item.sourceOffset,
          duration: placement.item.duration,
          isEnabled: placement.item.isEnabled,
          isTrackMuted: model.visualTrackMuted,
          isSelected: model.selectedTimelineItem == .visual(placement.item.id),
          metrics: metrics,
          height: TimelineChrome.visualLaneHeight,
          showsTrimHandles: model.selectedTimelineItem == .visual(placement.item.id)
            && visualMove == nil,
          onTrimChanged: { edge, translation in
            trimVisual(placement, edge: edge, translation: translation)
          },
          onTrimEnded: { visualTrim = nil },
          onMoved: { translation in
            dragVisual(placement, translation: translation)
          },
          onMoveEnded: endVisualMove
        )
        .offset(x: metrics.x(for: placement.startTime) + (isMoving ? (visualMove?.translation ?? 0) : 0))
        .zIndex(isMoving ? 12 : (placement.item.transition != nil ? 2 : 0))
        .opacity(isMoving ? 0.92 : 1)
        .onTapGesture {
          model.selectedTimelineItem = .visual(placement.item.id)
        }
        .overlay {
          GeometryReader { proxy in
            SecondaryClickProbe { local in
              presentClipMenu(
                .visual(placement.item.id),
                at: CGPoint(
                  x: proxy.frame(in: .named(editorSpace)).minX + local.x,
                  y: proxy.frame(in: .named(editorSpace)).minY + local.y
                )
              )
            }
          }
        }
        if visualMove == nil, model.project.timeline.canApplyVisualTransition(placement.item.id) {
          let overlap = model.project.timeline.transitionOverlap(of: placement.item.id)
          let hasTransition = placement.item.transition != nil
          TimelineTransitionHandle(
            duration: overlap,
            kind: placement.item.transition?.kind,
            isSelected: model.showsTransitionsPanel
              && model.transitionHostClip?.id == placement.item.id,
            metrics: metrics,
            laneHeight: TimelineChrome.visualLaneHeight,
            onSelect: {
              clipMenu = nil
              appendMenu = nil
              model.openTransitions(for: placement.item.id)
            },
            onContextMenu: { origin in
              presentClipMenu(.visual(placement.item.id), at: origin)
            },
            onDurationChanged: { translation in
              dragTransition(placement.item, translation: translation)
            },
            onDurationEnded: { transitionDrag = nil }
          )
          .offset(
            x: metrics.x(for: placement.startTime)
              - (hasTransition ? 0 : TimelineTransitionHandle.buttonSize / 2),
            y: hasTransition
              ? 0
              : (TimelineChrome.visualLaneHeight - TimelineTransitionHandle.buttonSize) / 2
          )
          .zIndex(10)
        }
        }
      }
      if let visualMove, visualMove.proposedIndex != visualMove.fromIndex {
        insertionCaret(
          x: insertionX(
            dest: visualMove.proposedIndex,
            placements: visualMove.reorderTargets
          ),
          height: TimelineChrome.visualLaneHeight
        )
      }
      appendControl(isVisual: true)
        .offset(
          x: metrics.x(for: model.project.timeline.visualDuration) + 8,
          y: (TimelineChrome.visualLaneHeight - TimelineChrome.appendButtonSize) / 2
        )
    }
    .frame(height: TimelineChrome.visualLaneHeight)
    .overlay {
      if model.visualTrackMuted {
        Color.black.opacity(0.16).allowsHitTesting(false)
      }
    }
    .timelineMediaDrop(lane: .visual, model: model, accessibilityID: "visual-lane-drop")
  }

  private var audioLane: some View {
    let renderedItems = model.project.timeline.audioItems.filter { item in
      audioMove?.itemID == item.id
        || shouldRenderTimelineSpan(start: item.startTime, duration: item.duration)
    }
    return ZStack(alignment: .topLeading) {
      laneBackground(alt: true)
      ForEach(renderedItems) { item in
        let isMoving = audioMove?.itemID == item.id
        TimelineClipView(
          title: model.project.asset(id: item.assetID)?.displayName ?? "Audio",
          kind: .audio,
          duration: item.duration,
          isEnabled: item.isEnabled,
          isTrackMuted: model.audioTrackMuted,
          isSelected: model.selectedTimelineItem == .audio(item.id),
          metrics: metrics,
          height: TimelineChrome.audioLaneHeight,
          showsTrimHandles: model.selectedTimelineItem == .audio(item.id)
            && audioMove == nil,
          onTrimChanged: { edge, translation in
            trimAudio(item, edge: edge, translation: translation)
          },
          onTrimEnded: { audioTrim = nil },
          onMoved: { translation in
            dragAudio(item, translation: translation)
          },
          onMoveEnded: endAudioMove
        )
        .offset(x: metrics.x(for: item.startTime) + (isMoving ? (audioMove?.translation ?? 0) : 0))
        .zIndex(isMoving ? 12 : 0)
        .opacity(isMoving ? 0.92 : 1)
        .onTapGesture {
          model.selectedTimelineItem = .audio(item.id)
        }
        .overlay {
          GeometryReader { proxy in
            SecondaryClickProbe { local in
              presentClipMenu(
                .audio(item.id),
                at: CGPoint(
                  x: proxy.frame(in: .named(editorSpace)).minX + local.x,
                  y: proxy.frame(in: .named(editorSpace)).minY + local.y
                )
              )
            }
          }
        }
      }
      if let audioMove, audioMove.proposedIndex != audioMove.fromIndex {
        insertionCaret(
          x: insertionX(
            dest: audioMove.proposedIndex,
            placements: audioMove.reorderTargets
          ),
          height: TimelineChrome.audioLaneHeight
        )
      }
      appendControl(isVisual: false)
        .offset(
          x: metrics.x(for: model.project.timeline.audioTrackEnd) + 8,
          y: (TimelineChrome.audioLaneHeight - TimelineChrome.appendButtonSize) / 2
        )
    }
    .frame(height: TimelineChrome.audioLaneHeight)
    .overlay {
      if model.audioTrackMuted {
        Color.black.opacity(0.16).allowsHitTesting(false)
      }
    }
    .timelineMediaDrop(lane: .audio, model: model, accessibilityID: "audio-lane-drop")
  }

  private func effectLane(items: [EffectLaneItem]) -> some View {
    let renderedItems = items.filter { item in
      shouldRenderTimelineSpan(start: item.startTime, duration: item.duration)
    }
    return ZStack(alignment: .topLeading) {
      Color.white.opacity(0.045)
      ForEach(renderedItems) { item in
        TimelineEffectPill(
          title: item.effect.kind.label,
          tint: item.effect.kind.swatch,
          isSelected: model.selectedEffectID == item.effect.id,
          isEnabled: item.effect.isEnabled && item.clipEnabled,
          width: max(18, metrics.x(for: item.duration) - 4)
        )
        .offset(x: metrics.x(for: item.startTime), y: 3)
        .onTapGesture {
          model.openEffectSettings(clipID: item.clipID, effectID: item.effect.id)
        }
      }
    }
    .frame(height: TimelineChrome.effectLaneHeight)
    .overlay(alignment: .bottom) {
      Rectangle().fill(H3Color.hairSoft).frame(height: 1)
    }
    .overlay {
      if model.visualTrackMuted {
        Color.black.opacity(0.16).allowsHitTesting(false)
      }
    }
  }

  private var collapsedScrubber: some View {
    VStack(alignment: .leading, spacing: 7) {
      GeometryReader { proxy in
        let width = max(proxy.size.width, 1)
        ZStack(alignment: .topLeading) {
          ForEach([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], id: \.self) { fraction in
            Text(ProgramClock.formatShort(contentDuration * fraction))
              .font(.system(size: 9, design: .monospaced))
              .foregroundStyle(H3Color.textSecondary.opacity(0.7))
              .offset(x: collapsedX(fraction, width: width, edge: fraction))
          }
        }
      }
      .frame(height: 13)

      GeometryReader { proxy in
        let width = max(proxy.size.width, 1)
        ZStack(alignment: .topLeading) {
          collapsedLane(isAudio: false, width: width)
          collapsedLane(isAudio: true, width: width)
            .offset(y: 19)
          collapsedPlayhead(width: width)
        }
        .contentShape(Rectangle())
        .gesture(
          DragGesture(minimumDistance: 0)
            .onChanged { value in
              let fraction = min(max(value.location.x / width, 0), 1)
              model.seekPlayback(contentDuration * Double(fraction))
            }
        )
      }
      .frame(height: 34)
    }
    .padding(.horizontal, 16)
    .padding(.top, 13)
    .padding(.bottom, 17)
  }

  private func collapsedLane(isAudio: Bool, width: CGFloat) -> some View {
    ZStack(alignment: .leading) {
      RoundedRectangle(cornerRadius: 4, style: .continuous)
        .fill(Color.white.opacity(0.06))
        .overlay {
          RoundedRectangle(cornerRadius: 4, style: .continuous)
            .stroke(H3Color.hairSoft, lineWidth: 1)
        }
      if isAudio {
        ForEach(model.project.timeline.audioItems) { item in
          collapsedSegment(
            start: item.startTime,
            duration: item.duration,
            color: H3Color.clipAudio,
            width: width,
            isMuted: model.audioTrackMuted || !item.isEnabled
          )
        }
      } else {
        ForEach(visualPlacements, id: \.item.id) { placement in
          collapsedSegment(
            start: placement.startTime,
            duration: placement.item.duration,
            color: H3Color.clipVideo,
            width: width,
            isMuted: model.visualTrackMuted || !placement.item.isEnabled
          )
        }
      }
    }
    .frame(height: 15)
    .timelineMediaDrop(
      lane: isAudio ? .audio : .visual,
      model: model,
      accessibilityID: isAudio ? "audio-collapsed-drop" : "visual-collapsed-drop"
    )
  }

  private func collapsedSegment(
    start: TimeInterval,
    duration: TimeInterval,
    color: Color,
    width: CGFloat,
    isMuted: Bool
  ) -> some View {
    let span = max(contentDuration, 0.001)
    return Rectangle()
      .fill(color.opacity(isMuted ? 0.32 : 0.82))
      .saturation(isMuted ? 0.4 : 1)
      .frame(width: max(2, width * CGFloat(duration / span)))
      .offset(x: width * CGFloat(start / span))
  }

  private func collapsedPlayhead(width: CGFloat) -> some View {
    let span = max(contentDuration, 0.001)
    let x = width * CGFloat(model.playback.clock.currentTime / span)
    return Rectangle()
      .fill(H3Color.accent)
      .frame(width: 2)
      .padding(.vertical, -4)
      .overlay(alignment: .top) {
        Circle()
          .fill(H3Color.accent)
          .frame(width: 14, height: 14)
          .offset(y: -10)
      }
      .shadow(color: H3Color.accent.opacity(0.7), radius: 4)
      .offset(x: x)
      .allowsHitTesting(false)
  }

  private var playhead: some View {
    ZStack(alignment: .top) {
      Rectangle()
        .fill(H3Color.accent)
        .frame(width: 2)
        .frame(maxHeight: .infinity)
        .allowsHitTesting(false)
      PlayheadChevron()
        .fill(H3Color.accent)
        .frame(width: 15, height: 17)
        .shadow(color: .black.opacity(0.5), radius: 2, y: 1)
        .contentShape(Rectangle())
        .gesture(
          DragGesture(minimumDistance: 0)
            .onChanged { value in
              if playheadDragOrigin == nil {
                playheadDragOrigin = model.playback.clock.currentTime
              }
              if let origin = playheadDragOrigin {
                model.seekPlayback(origin + metrics.time(for: value.translation.width))
              }
            }
            .onEnded { _ in
              playheadDragOrigin = nil
            }
        )
    }
    .shadow(color: H3Color.accent.opacity(0.7), radius: 4)
    .offset(x: metrics.x(for: model.playback.clock.currentTime) - 7.5)
    .zIndex(6)
  }

  private func appendControl(isVisual: Bool) -> some View {
    let trackMuted = isVisual ? model.visualTrackMuted : model.audioTrackMuted
    return Button {
      toggleAppendMenu(isVisual: isVisual)
    } label: {
      appendPlus
    }
    .buttonStyle(.plain)
    .disabled(trackMuted)
    .opacity(trackMuted ? 0.34 : 1)
    .background {
      GeometryReader { proxy in
        Color.clear.preference(
          key: AppendButtonFrameKey.self,
          value: [
            isVisual ? "visual" : "audio": proxy.frame(in: .named(editorSpace)),
          ]
        )
      }
    }
    .onPreferenceChange(AppendButtonFrameKey.self) { frames in
      appendButtonFrames.merge(frames) { _, new in new }
    }
    .help(
      trackMuted
        ? "Enable the track to append"
        : (isVisual ? "Append visual" : "Append audio")
    )
    .accessibilityLabel(isVisual ? "Append visual" : "Append audio")
    .accessibilityIdentifier(isVisual ? "append-visual" : "append-audio")
  }

  private func toggleAppendMenu(isVisual: Bool) {
    clipMenu = nil
    if appendMenu?.isVisual == isVisual {
      appendMenu = nil
      return
    }
    let key = isVisual ? "visual" : "audio"
    let frame = appendButtonFrames[key] ?? .zero
    let menuHeight: CGFloat = isVisual ? 154 : 122
    appendMenu = AppendMenuPlacement(
      isVisual: isVisual,
      origin: CGPoint(x: frame.minX, y: frame.minY - menuHeight - 8)
    )
  }

  private var appendPlus: some View {
    Image(systemName: "plus")
      .font(.system(size: 16, weight: .semibold))
      .foregroundStyle(H3Color.textSecondary)
      .frame(width: TimelineChrome.appendButtonSize, height: TimelineChrome.appendButtonSize)
      .background(H3Color.controlFill)
      .overlay {
        RoundedRectangle(cornerRadius: 9, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
      .contentShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
  }

  private func presentClipMenu(_ target: ClipMenuPlacement.Target, at origin: CGPoint) {
    appendMenu = nil
    switch target {
    case .visual(let id):
      model.selectedTimelineItem = .visual(id)
    case .audio(let id):
      model.selectedTimelineItem = .audio(id)
    }
    clipMenu = ClipMenuPlacement(target: target, origin: origin)
  }

  private func dragTransition(_ item: VisualItem, translation: CGFloat) {
    if transitionDrag?.itemID != item.id {
      model.registerUndoCheckpoint()
      transitionDrag = TransitionDragSession(
        itemID: item.id,
        origin: item.transition?.duration ?? VisualTransitionMath.defaultDuration
      )
    }
    guard let transitionDrag else { return }
    if item.transition == nil {
      model.setVisualTransition(item.id, kind: .dissolve, registersUndo: false)
    }
    model.setVisualTransitionDuration(
      item.id,
      duration: max(0, transitionDrag.origin + metrics.time(for: translation))
    )
  }

  private var visualPlacements: [VisualPlacement] {
    model.project.timeline.visualPlacements
  }

  private var orderedAudio: [AudioItem] {
    model.project.timeline.audioItems.sorted(by: TimelineReorderMath.audioOrder)
  }

  private var renderedTimeRange: ClosedRange<TimeInterval> {
    TimelineRuler.visibleTimeRange(
      scrollOffset: Double(timelineScrollX),
      viewportWidth: Double(timelineViewportWidth),
      pointsPerSecond: Double(metrics.pointsPerSecond),
      overscanPoints: Double(timelineViewportWidth)
    )
  }

  private func shouldRenderTimelineSpan(
    start: TimeInterval,
    duration: TimeInterval
  ) -> Bool {
    TimelineRuler.spanIntersects(
      start: start,
      duration: duration,
      visibleRange: renderedTimeRange
    )
  }

  private func dragVisual(_ placement: VisualPlacement, translation: CGFloat) {
    if visualMove?.itemID != placement.item.id {
      clipMenu = nil
      appendMenu = nil
      model.selectedTimelineItem = .visual(placement.item.id)
      let placements = visualPlacements
      let from = placements.firstIndex(where: { $0.item.id == placement.item.id }) ?? 0
      visualMove = ClipMoveSession(
        itemID: placement.item.id,
        fromIndex: from,
        originStart: placement.startTime,
        translation: 0,
        proposedIndex: from,
        reorderTargets: placements
          .filter { $0.item.id != placement.item.id }
          .map { (start: $0.startTime, duration: $0.item.duration) }
      )
      NSCursor.closedHand.push()
    }
    let dropTime =
      placement.startTime + placement.item.duration / 2 + metrics.time(for: translation)
    guard var session = visualMove else { return }
    session.translation = translation
    session.proposedIndex = TimelineReorderMath.destinationIndex(
      dropTime: dropTime,
      others: session.reorderTargets
    )
    visualMove = session
  }

  private func endVisualMove() {
    defer {
      if visualMove != nil {
        NSCursor.pop()
      }
      visualMove = nil
    }
    guard let visualMove else { return }
    if visualMove.proposedIndex != visualMove.fromIndex {
      model.moveVisual(visualMove.itemID, toIndex: visualMove.proposedIndex)
    }
  }

  private func dragAudio(_ item: AudioItem, translation: CGFloat) {
    if audioMove?.itemID != item.id {
      clipMenu = nil
      appendMenu = nil
      model.selectedTimelineItem = .audio(item.id)
      let ordered = orderedAudio
      let from = ordered.firstIndex(where: { $0.id == item.id }) ?? 0
      audioMove = ClipMoveSession(
        itemID: item.id,
        fromIndex: from,
        originStart: item.startTime,
        translation: 0,
        proposedIndex: from,
        reorderTargets: ordered
          .filter { $0.id != item.id }
          .map { (start: $0.startTime, duration: $0.duration) }
      )
      NSCursor.closedHand.push()
    }
    let dropTime = item.startTime + item.duration / 2 + metrics.time(for: translation)
    guard var session = audioMove else { return }
    session.translation = translation
    session.proposedIndex = TimelineReorderMath.destinationIndex(
      dropTime: dropTime,
      others: session.reorderTargets
    )
    audioMove = session
  }

  private func endAudioMove() {
    defer {
      if audioMove != nil {
        NSCursor.pop()
      }
      audioMove = nil
    }
    guard let audioMove else { return }
    if audioMove.proposedIndex != audioMove.fromIndex {
      model.moveAudio(audioMove.itemID, toIndex: audioMove.proposedIndex)
    } else {
      model.setAudioStart(
        audioMove.itemID,
        startTime: audioMove.originStart + metrics.time(for: audioMove.translation)
      )
    }
  }

  private func insertionX(
    dest: Int,
    placements: [(start: TimeInterval, duration: TimeInterval)]
  ) -> CGFloat {
    if dest <= 0 {
      return placements.first.map { metrics.x(for: $0.start) } ?? 0
    }
    if dest >= placements.count {
      return placements.last.map { metrics.x(for: $0.start + $0.duration) } ?? 0
    }
    return metrics.x(for: placements[dest].start)
  }

  private func insertionCaret(x: CGFloat, height: CGFloat) -> some View {
    RoundedRectangle(cornerRadius: 1, style: .continuous)
      .fill(H3Color.accent)
      .frame(width: 2, height: height - 4)
      .offset(x: x - 1, y: 2)
      .shadow(color: H3Color.accent.opacity(0.7), radius: 3)
      .zIndex(14)
      .allowsHitTesting(false)
  }

  private func trimVisual(
    _ placement: VisualPlacement,
    edge: VisualTrimEdge,
    translation: CGFloat
  ) {
    model.selectedTimelineItem = .visual(placement.item.id)
    if visualTrim?.itemID != placement.item.id || visualTrim?.edge != edge {
      model.registerUndoCheckpoint()
      let asset = model.project.asset(id: placement.item.assetID)
      visualTrim = VisualTrimSession(
        itemID: placement.item.id,
        edge: edge,
        startTime: placement.startTime,
        origin: VisualTrim(
          duration: placement.item.duration,
          sourceOffset: placement.item.sourceOffset,
          gapBefore: placement.item.gapBefore
        ),
        sourceLimit: VisualTrimMath.sourceLimit(
          kind: asset?.kind ?? .video,
          sourceDuration: asset?.duration ?? placement.item.duration
        )
      )
    }
    guard let visualTrim else { return }
    model.applyVisualTrim(
      visualTrim.itemID,
      edge: visualTrim.edge,
      delta: metrics.time(for: translation),
      origin: visualTrim.origin,
      startTime: visualTrim.startTime,
      sourceLimit: visualTrim.sourceLimit
    )
  }

  private func trimAudio(
    _ item: AudioItem,
    edge: TimelineTrimEdge,
    translation: CGFloat
  ) {
    model.selectedTimelineItem = .audio(item.id)
    if audioTrim?.itemID != item.id || audioTrim?.edge != edge {
      model.registerUndoCheckpoint()
      let asset = model.project.asset(id: item.assetID)
      let bounds = model.project.timeline.audioNeighborBounds(of: item.id)
      audioTrim = AudioTrimSession(
        itemID: item.id,
        edge: edge,
        origin: AudioTrim(
          startTime: item.startTime,
          duration: item.duration,
          sourceOffset: item.sourceOffset
        ),
        sourceLimit: asset?.duration ?? item.duration,
        earliestStart: bounds.earliestStart,
        latestEnd: bounds.latestEnd
      )
    }
    guard let audioTrim else { return }
    model.applyAudioTrim(
      audioTrim.itemID,
      edge: audioTrim.edge,
      delta: metrics.time(for: translation),
      origin: audioTrim.origin,
      sourceLimit: audioTrim.sourceLimit,
      earliestStart: audioTrim.earliestStart,
      latestEnd: audioTrim.latestEnd
    )
  }

  private func laneBackground(alt: Bool) -> some View {
    (alt ? Color.white.opacity(0.045) : Color.clear)
      .overlay(alignment: .bottom) {
        Rectangle().fill(H3Color.hairSoft).frame(height: 1)
      }
  }

  private func collapsedX(_ fraction: Double, width: CGFloat, edge: Double) -> CGFloat {
    if edge == 0 { return 0 }
    if edge == 1 { return width - 28 }
    return width * CGFloat(fraction) - 12
  }
}

private struct VisualTrimSession {
  var itemID: UUID
  var edge: TimelineTrimEdge
  var startTime: TimeInterval
  var origin: VisualTrim
  var sourceLimit: TimeInterval?
}

private struct AudioTrimSession {
  var itemID: UUID
  var edge: TimelineTrimEdge
  var origin: AudioTrim
  var sourceLimit: TimeInterval
  var earliestStart: TimeInterval
  var latestEnd: TimeInterval
}

private struct ClipMoveSession {
  var itemID: UUID
  var fromIndex: Int
  var originStart: TimeInterval
  var translation: CGFloat
  var proposedIndex: Int
  var reorderTargets: [(start: TimeInterval, duration: TimeInterval)]
}

private struct TransitionDragSession {
  var itemID: UUID
  var origin: TimeInterval
}

private struct AppendButtonFrameKey: PreferenceKey {
  static let defaultValue: [String: CGRect] = [:]

  static func reduce(value: inout [String: CGRect], nextValue: () -> [String: CGRect]) {
    value.merge(nextValue()) { _, new in new }
  }
}



private struct PlayheadChevron: Shape {
  func path(in rect: CGRect) -> Path {
    var path = Path()
    path.move(to: CGPoint(x: rect.minX, y: rect.minY))
    path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
    path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + rect.height * 0.6))
    path.addLine(to: CGPoint(x: rect.midX, y: rect.maxY))
    path.addLine(to: CGPoint(x: rect.minX, y: rect.minY + rect.height * 0.6))
    path.closeSubpath()
    return path
  }
}
