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
        HStack(spacing: 0) {
          if model.showsProjectSettings {
            ProjectSettingsPanel(model: model)
          } else if model.showsTransitionsPanel {
            TransitionsPanelView(model: model)
          } else if model.showsEffectsPanel {
            EffectsPanelView(model: model)
          }
          VStack(spacing: 0) {
            ProgramCanvasView(model: model, clipMenu: $clipMenu)
              .frame(maxWidth: .infinity, maxHeight: .infinity)
            VStack(spacing: 0) {
              TransportBarView(model: model)
              ProgramTimelineView(
                model: model,
                appendMenu: $appendMenu,
                clipMenu: $clipMenu
              )
            }
          }
        }
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
    .animation(.easeOut(duration: 0.16), value: model.showsProjectSettings)
    .animation(.easeOut(duration: 0.16), value: model.showsExport)
    .sheet(isPresented: $model.showsModelSettings) {
      ModelSettingsView(model: model)
    }
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
    .onKeyPress(.escape) {
      if clipMenu != nil || appendMenu != nil {
        clipMenu = nil
        appendMenu = nil
        return .handled
      }
      if model.showsEffectsPanel {
        model.showsEffectsPanel = false
        model.selectedEffectID = nil
        return .handled
      }
      if model.showsTransitionsPanel {
        model.closeTransitionsPanel()
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

  private let editorSpace = "editor-root"

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
          trackName: appendMenu.isVisual ? "V1" : "A1",
          appendTime: appendMenu.isVisual
            ? model.project.timeline.visualDuration
            : model.project.timeline.audioTrackEnd,
          items: appendMenu.isVisual
            ? TimelineAppendMenu.visualItems()
            : TimelineAppendMenu.audioItems()
        ) { item in
          let importingVisual = appendMenu.isVisual
          self.appendMenu = nil
          self.clipMenu = nil
          switch item.action {
          case .generate(let kind):
            model.presentGeneration(kind)
          case .importFiles:
            presentImportPanel(ontoVisualLane: importingVisual)
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
      case .toggleNativeAudio, .coverCanvas, .fitToCanvas, .rotate, .resetTransform:
        break
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
        Text("H3ddle")
          .font(.system(size: 15, weight: .semibold))
          .accessibilityIdentifier("editor-root")

        Button {
          if model.showsProjectSettings {
            model.showsProjectSettings = false
          } else {
            model.showsEffectsPanel = false
            model.closeTransitionsPanel()
            model.showsProjectSettings = true
          }
        } label: {
          Image(systemName: "slider.vertical.3")
        }
        .buttonStyle(H3IconButtonStyle(isActive: model.showsProjectSettings, size: 30))
        .help(model.showsProjectSettings ? "Close Project panel" : "Open Project panel")
        .accessibilityIdentifier("project-settings-toggle")
      }

      Spacer(minLength: 0)

      Button {
        model.showsModelSettings = true
      } label: {
        HStack(spacing: 6) {
          Circle()
            .fill(model.isGenerating ? H3Color.accent : modelStatusColor)
            .frame(width: 6, height: 6)
          Text("Models")
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(H3Color.textSecondary)
        }
      }
      .buttonStyle(H3QuietButtonStyle())
      .help(model.modelStatusTitle)
      .accessibilityIdentifier("model-status")

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

  private var modelStatusColor: Color {
    switch model.modelValidationState {
    case .ready, .validating:
      H3Color.accent
    case .failed:
      H3Color.danger
    case .notSelected:
      H3Color.textSecondary.opacity(0.45)
    }
  }
}
