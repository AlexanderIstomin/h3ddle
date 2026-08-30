import AppKit
import H3ddleCore
import H3ddleDesignSystem
import H3ddleGeneration
import H3ddleMedia
import SwiftUI

struct EditorView: View {
  @Bindable var model: AppModel
  @State private var appendMenu: AppendMenuPlacement?
  @State private var clipMenu: ClipMenuPlacement?

  var body: some View {
    ZStack {
      VStack(spacing: 0) {
        toolbar
        Divider().overlay(H3Color.line)
        workspace
        programDock
      }
      .background(H3Color.canvas)
      .foregroundStyle(H3Color.textPrimary)
      .coordinateSpace(name: editorSpace)
      .overlay(alignment: .topLeading) {
        appendMenuOverlay
        clipMenuOverlay
      }

      if let kind = model.activeGenerationKind {
        GenerationStudioView(model: model, kind: kind)
          .transition(.opacity)
      }

      if model.showsExport {
        ExportModalView(model: model)
          .transition(.opacity)
      }
    }
    .animation(.easeOut(duration: 0.16), value: model.activeGenerationKind)
    .animation(.easeOut(duration: 0.16), value: model.openPanel)
    .animation(.easeOut(duration: 0.16), value: model.showsExport)
    .alert(
      "Couldn’t import media",
      isPresented: Binding(
        get: { model.importErrorMessage != nil },
        set: { if !$0 { model.importErrorMessage = nil } }
      )
    ) {
      Button("OK", role: .cancel) { model.importErrorMessage = nil }
    } message: {
      Text(model.importErrorMessage ?? "")
    }
    .onKeyPress(.space) {
      guard model.activeGenerationKind == nil, !model.showsExport else { return .ignored }
      model.togglePlayback()
      return .handled
    }
    .onKeyPress(characters: CharacterSet(charactersIn: "sS")) { _ in
      guard model.activeGenerationKind == nil, !model.showsExport else { return .ignored }
      guard model.canSplitSelectedAtPlayhead else { return .ignored }
      model.splitSelectedAtPlayhead()
      return .handled
    }
    .onKeyPress(keys: ["z", "Z"], phases: .down) { press in
      guard press.modifiers.contains(.command) else { return .ignored }
      guard model.activeGenerationKind == nil, !model.showsExport else { return .ignored }
      if NSApp.keyWindow?.firstResponder is NSTextView { return .ignored }
      if press.modifiers.contains(.shift) {
        guard model.canRedo else { return .ignored }
        model.redo()
      } else {
        guard model.canUndo else { return .ignored }
        model.undo()
      }
      return .handled
    }
    .onKeyPress(keys: ["d", "D"], phases: .down) { press in
      guard press.modifiers.contains(.command) else { return .ignored }
      guard model.activeGenerationKind == nil, !model.showsExport else { return .ignored }
      guard model.selectedTimelineItem != nil else { return .ignored }
      model.duplicateSelectedTimelineItem()
      return .handled
    }
    .onKeyPress(.delete) {
      guard model.activeGenerationKind == nil, !model.showsExport else { return .ignored }
      guard model.selectedTimelineItem != nil else { return .ignored }
      model.deleteSelectedTimelineItem()
      return .handled
    }
    .onKeyPress(.deleteForward) {
      guard model.activeGenerationKind == nil, !model.showsExport else { return .ignored }
      guard model.selectedTimelineItem != nil else { return .ignored }
      model.deleteSelectedTimelineItem()
      return .handled
    }
    .onKeyPress(keys: ["t", "T"], phases: .down) { press in
      guard press.modifiers.contains(.command) else { return .ignored }
      guard model.activeGenerationKind == nil, !model.showsExport else { return .ignored }
      if NSApp.keyWindow?.firstResponder is NSTextView { return .ignored }
      if case .text = model.selectedTimelineItem {
        model.openTextPanel()
      } else {
        model.insertTextAtPlayhead(opensInspector: true)
      }
      return .handled
    }
    .onKeyPress(.escape) {
      if clipMenu != nil || appendMenu != nil {
        clipMenu = nil
        appendMenu = nil
        return .handled
      }
      if model.openPanel != nil {
        model.closeOpenPanel()
        return .handled
      }
      return .ignored
    }
    .onChange(of: model.activeGenerationKind) { _, _ in
      clipMenu = nil
      appendMenu = nil
    }
    .onChange(of: model.showsExport) { _, _ in
      clipMenu = nil
      appendMenu = nil
    }
  }

  private var workspace: some View {
    ZStack(alignment: .leading) {
      HStack(spacing: 0) {
        Color.clear.frame(width: LeftRailMetrics.width)
        ProgramCanvasView(model: model, clipMenu: $clipMenu)
          .frame(maxWidth: .infinity, maxHeight: .infinity)
      }
      if let panel = model.openPanel {
        LeftRailPanel(
          title: panel.panelTitle,
          navigationHelp: panel.parentPanel.map { "Back to \($0.panelTitle)" } ?? "Collapse panel",
          onClose: { navigateBack(from: panel) }
        ) {
          panelBody(panel)
        }
        .padding(.leading, LeftRailMetrics.width)
        .transition(.move(edge: .leading).combined(with: .opacity))
        .zIndex(1)
      }
      LeftRailView(model: model)
        .zIndex(2)
    }
  }

  private var programDock: some View {
    VStack(spacing: 0) {
      TransportBarView(model: model)
      ProgramTimelineView(
        model: model,
        appendMenu: $appendMenu,
        clipMenu: $clipMenu
      )
    }
    .frame(maxWidth: .infinity)
  }

  @ViewBuilder
  private func panelBody(_ panel: EditorPanel) -> some View {
    switch panel {
    case .project:
      ProjectSettingsPanel(model: model, embedded: true)
    case .video:
      LibraryPanelView(model: model, kind: .video)
    case .images:
      LibraryPanelView(model: model, kind: .image)
    case .audio:
      LibraryPanelView(model: model, kind: .audio)
    case .effects:
      EffectsPanelView(model: model, embedded: true)
    case .transitions:
      TransitionsPanelView(model: model, embedded: true)
    case .text:
      TextInspectorPanel(model: model, embedded: true)
    case .adjust:
      AdjustPanelView(model: model)
    case .upscale:
      UpscalePanelView(model: model)
    case .queue:
      GenerationQueueView(model: model, queue: model.generationQueue, embedded: true)
        .id(model.generationQueueRevision)
    case .models:
      ModelSettingsView(model: model, embedded: true)
    }
  }

  private func navigateBack(from panel: EditorPanel) {
    if let parent = panel.parentPanel {
      model.openRail(parent)
    } else {
      model.closeOpenPanel()
    }
  }

  private let editorSpace = "editor-root"

  private func appendTrackName(_ track: AppendMenuPlacement.Track) -> String {
    switch track {
    case .visual: "V1"
    case .audio: "A1"
    case .text: "T1"
    }
  }

  private func appendTime(_ track: AppendMenuPlacement.Track) -> TimeInterval {
    switch track {
    case .visual: model.project.timeline.visualDuration
    case .audio: model.project.timeline.audioTrackEnd
    case .text: model.playback.clock.currentTime
    }
  }

  private func appendItems(_ track: AppendMenuPlacement.Track) -> [TimelineAppendMenuItem] {
    switch track {
    case .visual: TimelineAppendMenu.visualItems()
    case .audio: TimelineAppendMenu.audioItems()
    case .text: TimelineAppendMenu.textItems()
    }
  }

  private func presentImportPanel(ontoVisualLane: Bool) {
    let panel = NSOpenPanel()
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = true
    panel.allowedContentTypes =
      ontoVisualLane ? MediaImport.visualContentTypes : MediaImport.audioContentTypes
    panel.prompt = "Add"
    panel.message =
      ontoVisualLane
      ? "Add video or image files to V1"
      : "Add audio files to A1"
    guard panel.runModal() == .OK, !panel.urls.isEmpty else { return }
    Task {
      await model.importFiles(panel.urls, onto: ontoVisualLane ? .visual : .audio)
    }
  }

  @ViewBuilder
  private var appendMenuOverlay: some View {
    if let appendMenu {
      ZStack(alignment: .topLeading) {
        Color.clear
          .contentShape(Rectangle())
          .onTapGesture {
            self.appendMenu = nil
          }
        TimelineAppendMenu(
          trackName: appendTrackName(appendMenu.track),
          appendTime: appendTime(appendMenu.track),
          items: appendItems(appendMenu.track)
        ) { item in
          let importingVisual = appendMenu.track == .visual
          self.appendMenu = nil
          self.clipMenu = nil
          switch item.action {
          case .generate(let kind):
            model.presentGeneration(kind)
          case .importFiles:
            presentImportPanel(ontoVisualLane: importingVisual)
          case .addText:
            model.insertTextAtPlayhead(opensInspector: true)
          }
        }
        .offset(x: appendMenu.origin.x, y: appendMenu.origin.y)
      }
    }
  }

  @ViewBuilder
  private var clipMenuOverlay: some View {
    if let clipMenu, let content = clipMenuContent(for: clipMenu.target) {
      GeometryReader { proxy in
        ZStack(alignment: .topLeading) {
          Color.clear
            .contentShape(Rectangle())
            .onTapGesture {
              self.clipMenu = nil
            }
            .overlay {
              SecondaryClickProbe { _ in
                self.clipMenu = nil
              }
            }
          TimelineClipMenu(title: content.title, items: content.items) { item in
            performClipMenu(item, target: clipMenu.target)
          }
          .offset(
            clipMenuOffset(
              clipMenu.origin,
              itemCount: content.items.count,
              in: proxy.size
            )
          )
        }
      }
    }
  }

  private func clipMenuContent(
    for target: ClipMenuPlacement.Target
  ) -> (title: String, items: [TimelineClipMenuItem])? {
    switch target {
    case .visual(let id):
      guard let item = model.project.timeline.visualItems.first(where: { $0.id == id }) else {
        return nil
      }
      let asset = model.project.asset(id: item.assetID)
      return (
        asset?.displayName ?? "Visual",
        TimelineClipMenu.visualItems(
          item: item,
          kind: asset?.kind ?? .video,
          canSplit: model.canSplit(.visual(id))
        )
      )
    case .audio(let id):
      guard let item = model.project.timeline.audioItems.first(where: { $0.id == id }) else {
        return nil
      }
      return (
        model.project.asset(id: item.assetID)?.displayName ?? "Audio",
        TimelineClipMenu.audioItems(item: item, canSplit: model.canSplit(.audio(id)))
      )
    case .text(let id):
      guard let item = model.project.timeline.textItems.first(where: { $0.id == id }) else {
        return nil
      }
      let title = item.text.split(whereSeparator: \.isNewline).first.map(String.init) ?? "Text"
      return (
        title.isEmpty ? "Text" : title,
        TimelineClipMenu.textItems(item: item, canSplit: model.canSplit(.text(id)))
      )
    case .insertText:
      return ("T1", TimelineClipMenu.addTextItems())
    }
  }

  private func performClipMenu(_ item: TimelineClipMenuItem, target: ClipMenuPlacement.Target) {
    clipMenu = nil
    guard let action = item.action else { return }
    switch target {
    case .visual(let id):
      model.selectedTimelineItem = .visual(id)
      switch action {
      case .duplicate:
        model.duplicateVisual(id)
      case .toggleEnabled:
        model.toggleVisual(id)
      case .toggleNativeAudio:
        model.toggleVisualNativeAudio(id)
      case .split:
        model.split(.visual(id))
      case .coverCanvas:
        model.setVisualCanvasFit(id, .cover)
      case .fitToCanvas:
        model.setVisualCanvasFit(id, .fit)
      case .rotate:
        model.rotateVisual(id)
      case .resetTransform:
        model.resetVisualTransform(id)
      case .remove:
        model.removeVisual(id)
      case .addText:
        break
      }
    case .audio(let id):
      model.selectedTimelineItem = .audio(id)
      switch action {
      case .duplicate:
        model.duplicateAudio(id)
      case .toggleEnabled:
        model.toggleAudio(id)
      case .split:
        model.split(.audio(id))
      case .remove:
        model.removeAudio(id)
      case .toggleNativeAudio, .coverCanvas, .fitToCanvas, .rotate, .resetTransform, .addText:
        break
      }
    case .text(let id):
      model.selectedTimelineItem = .text(id)
      switch action {
      case .duplicate:
        model.duplicateText(id)
      case .toggleEnabled:
        model.toggleText(id)
      case .split:
        model.split(.text(id))
      case .resetTransform:
        model.resetTextTransform(id)
      case .remove:
        model.removeText(id)
      case .toggleNativeAudio, .coverCanvas, .fitToCanvas, .rotate, .addText:
        break
      }
    case .insertText:
      if action == .addText {
        model.insertTextAtPlayhead(opensInspector: true)
      }
    }
  }

  private func clipMenuOffset(_ origin: CGPoint, itemCount: Int, in size: CGSize) -> CGSize {
    let width: CGFloat = 208
    let estimatedHeight = 42 + CGFloat(itemCount) * 32
    let x = min(max(8, origin.x), max(8, size.width - width - 8))
    let y = min(max(8, origin.y), max(8, size.height - estimatedHeight - 8))
    return CGSize(width: x, height: y)
  }

  private var toolbar: some View {
    HStack(spacing: H3Spacing.medium) {
      HStack(spacing: 9) {
        Image(nsImage: NSApp.applicationIconImage)
          .resizable()
          .interpolation(.high)
          .frame(width: 24, height: 24)
          .accessibilityHidden(true)
        Text(model.project.name)
          .font(.system(size: 15, weight: .semibold))
          .accessibilityIdentifier("editor-root")

        Button {
          if model.showsProjectSettings {
            model.closeOpenPanel()
          } else {
            model.openRail(.project)
          }
        } label: {
          Image(systemName: "slider.vertical.3")
        }
        .buttonStyle(H3IconButtonStyle(isActive: model.showsProjectSettings, size: 30))
        .help(model.showsProjectSettings ? "Close Project panel" : "Open Project panel")
        .accessibilityIdentifier("project-settings-toggle")
      }

      Spacer(minLength: 0)

      Button("Export") {
        model.showsExport = true
      }
      .buttonStyle(H3PrimaryButtonStyle())
      .accessibilityIdentifier("export-button")
    }
    .padding(.horizontal, H3Spacing.medium)
    .frame(height: 50)
    .background(H3Color.chrome)
  }
}
