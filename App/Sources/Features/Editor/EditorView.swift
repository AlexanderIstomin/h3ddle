import H3ddleCore
import H3ddleDesignSystem
import H3ddleGeneration
import SwiftUI

struct EditorView: View {
  @Bindable var model: AppModel

  var body: some View {
    VStack(spacing: 0) {
      toolbar
      Divider().overlay(H3Color.line)
      preview
      Divider().overlay(H3Color.line)
      ProgramTimelineView(model: model)
        .frame(height: 244)
    }
    .background(H3Color.canvas)
    .foregroundStyle(H3Color.textPrimary)
    .sheet(item: $model.activeGenerationKind) { kind in
      GenerationStudioView(model: model, kind: kind)
        .interactiveDismissDisabled(model.isGenerating)
    }
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
              .foregroundStyle(H3Color.canvas)
          }
        Text("H3ddle")
          .font(.system(size: 15, weight: .semibold))
          .accessibilityIdentifier("editor-root")
      }

      Divider()
        .overlay(H3Color.line)
        .frame(height: 20)

      Text(model.project.name)
        .font(.system(size: 12, weight: .medium))
        .foregroundStyle(H3Color.textSecondary)

      Spacer()

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
      .buttonStyle(.plain)
      .accessibilityIdentifier("model-status")

      Button("Export") {
        model.showsExportNotice = true
      }
      .buttonStyle(H3PrimaryButtonStyle())
      .accessibilityIdentifier("export-button")
    }
    .padding(.horizontal, H3Spacing.medium)
    .frame(height: 52)
    .background(H3Color.surface)
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

  private var preview: some View {
    ZStack {
      Color.black

      if let asset = model.previewAsset {
        GeneratedAssetPreview(
          asset: asset,
          generationDuration: model.generationDurationDescription(for: asset)
        )
        .id(asset.id)
        .transition(.opacity)
      } else {
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
      }
    }
    .animation(.easeOut(duration: 0.2), value: model.previewAsset?.id)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .accessibilityIdentifier("program-preview")
  }
}
