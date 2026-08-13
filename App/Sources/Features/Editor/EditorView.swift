import H3ddleDesignSystem
import H3ddleGeneration
import SwiftUI

struct EditorView: View {
  @Bindable var model: AppModel
  @State private var appendMenu: AppendMenuPlacement?

  var body: some View {
    ZStack {
      VStack(spacing: 0) {
        toolbar
        Divider().overlay(H3Color.line)
        HStack(spacing: 0) {
          if model.showsProjectSettings {
            ProjectSettingsPanel(model: model)
          }
          VStack(spacing: 0) {
            ProgramCanvasView(model: model)
              .frame(maxWidth: .infinity, maxHeight: .infinity)
            VStack(spacing: 0) {
              TransportBarView(model: model)
              ProgramTimelineView(model: model, appendMenu: $appendMenu)
            }
          }
        }
      }
      .background(H3Color.canvas)
      .foregroundStyle(H3Color.textPrimary)
      .coordinateSpace(name: editorSpace)
      .overlay(alignment: .topLeading) {
        appendMenuOverlay
      }

      if let kind = model.activeGenerationKind {
        GenerationStudioView(model: model, kind: kind)
          .transition(.opacity)
      }
    }
    .animation(.easeOut(duration: 0.16), value: model.activeGenerationKind)
    .animation(.easeOut(duration: 0.16), value: model.showsProjectSettings)
    .sheet(isPresented: $model.showsModelSettings) {
      ModelSettingsView(model: model)
    }
    .alert("Export is being connected", isPresented: $model.showsExportNotice) {
      Button("OK", role: .cancel) {}
    } message: {
      Text(
        "The native AVFoundation export pipeline is reserved by the scaffold but not implemented in this slice."
      )
    }
    .onKeyPress(.space) {
      guard model.activeGenerationKind == nil else { return .ignored }
      model.togglePlayback()
      return .handled
    }
  }

  private let editorSpace = "editor-root"

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
          self.appendMenu = nil
          model.presentGeneration(item.kind)
        }
        .offset(x: appendMenu.origin.x, y: appendMenu.origin.y)
      }
    }
  }

  private var toolbar: some View {
    HStack(spacing: H3Spacing.medium) {
      HStack(spacing: 9) {
        RoundedRectangle(cornerRadius: 5, style: .continuous)
          .fill(H3Color.accent)
          .frame(width: 24, height: 24)
          .overlay {
            Text("H3")
              .font(.system(size: 9, weight: .black, design: .rounded))
              .foregroundStyle(Color.white)
          }
        Text("H3ddle")
          .font(.system(size: 15, weight: .semibold))
          .accessibilityIdentifier("editor-root")

        Button {
          model.showsProjectSettings.toggle()
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
          Text(model.isGenerating ? model.generationPhase : model.modelStatusTitle)
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(H3Color.textSecondary)
        }
      }
      .buttonStyle(H3QuietButtonStyle())
      .accessibilityIdentifier("model-status")

      Button("Export") {
        model.showsExportNotice = true
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
