import H3ddleDesignSystem
import H3ddleEngineProtocol
import H3ddleGeneration
import SwiftUI

struct GenerationStudioView: View {
  @Bindable var model: AppModel
  let kind: GenerationKind

  private static let h3FPS = 24.0
  private static let h3MinimumFrames = 22
  private static let h3FrameChunk = 17

  @State private var duration: Double = 1
  /// Index into H3's legal temporal shapes: 22 + 17n frames.
  @State private var alignedDurationStep: Double = 0
  @State private var quality: EngineGenerationQuality = .preview
  @State private var denoisingSteps: Double = Double(
    EngineGenerationQuality.preview.denoisingSteps
  )
  @State private var activeDiTLayers = EngineGenerationQuality.preview.activeDiTLayers
  @State private var coreReuse = 1

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      header
        .fixedSize(horizontal: false, vertical: true)
      Divider().overlay(H3Color.line)
      ScrollView {
        form
      }
      .scrollBounceBehavior(.basedOnSize)
      Divider().overlay(H3Color.line)
      footer
        .fixedSize(horizontal: false, vertical: true)
    }
    .frame(width: 600, height: 600)
    .background(H3Color.surface)
    .foregroundStyle(H3Color.textPrimary)
  }

  private var header: some View {
    HStack {
      VStack(alignment: .leading, spacing: 4) {
        Text("GENERATE \(kind.rawValue.uppercased())")
          .font(.system(size: 10, weight: .bold))
          .tracking(1.1)
          .foregroundStyle(H3Color.accent)
        Text(kind == .audio ? "Create audio for the program" : "Create the next visual")
          .font(.system(size: 18, weight: .semibold))
          .accessibilityIdentifier("generation-studio")
      }
      Spacer()
      Button {
        model.activeGenerationKind = nil
      } label: {
        Image(systemName: "xmark")
      }
      .buttonStyle(.plain)
      .disabled(model.isGenerating)
    }
    .padding(H3Spacing.large)
  }

  private var form: some View {
    VStack(alignment: .leading, spacing: H3Spacing.large) {
      VStack(alignment: .leading, spacing: H3Spacing.small) {
        Text("Prompt")
          .font(.system(size: 12, weight: .semibold))
        TextEditor(text: $model.generationPrompt)
          .font(.system(size: 13))
          .scrollContentBackground(.hidden)
          .padding(10)
          .background(H3Color.canvas)
          .overlay {
            RoundedRectangle(cornerRadius: H3Radius.medium, style: .continuous)
              .stroke(H3Color.line, lineWidth: 1)
          }
          .clipShape(RoundedRectangle(cornerRadius: H3Radius.medium, style: .continuous))
          .frame(height: 126)
          .accessibilityIdentifier("generation-prompt")
      }

      VStack(alignment: .leading, spacing: H3Spacing.small) {
        HStack {
          Text(kind == .image ? "Display duration" : "Duration")
            .font(.system(size: 12, weight: .semibold))
          Spacer()
          Text(
            kind == .video && model.nativeVideoGenerationIsReady
              ? String(format: "%.1f s · %d frames", alignedSeconds, alignedFrames)
              : String(format: "%.0f seconds", duration)
          )
            .font(.system(size: 11, weight: .medium, design: .monospaced))
            .foregroundStyle(H3Color.textSecondary)
        }
        if kind == .video, model.nativeVideoGenerationIsReady {
          // H3 only produces 22 + 17n frame clips; snapping the slider to
          // those shapes keeps the shortest (cheapest) clip reachable and
          // makes the shown duration the real one.
          Slider(value: $alignedDurationStep, in: 0...20, step: 1)
            .tint(H3Color.accent)
        } else {
          Slider(value: $duration, in: 1...15, step: 1)
            .tint(H3Color.accent)
        }
        if kind == .image, model.nativeImageGenerationIsReady {
          Text(
            "H3 has no native still mode. It generates one 22-frame chunk and keeps the last frame, the community still recipe."
          )
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .fixedSize(horizontal: false, vertical: true)
        }
        if kind == .video, model.nativeVideoGenerationIsReady, alignedSeconds > 3 {
          Text(
            "Long H3 clips increase transformer work sharply. Start with the shortest clip on M1-class Macs."
          )
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .fixedSize(horizontal: false, vertical: true)
        }
      }

      if usesNativeVisualSettings {
        VStack(alignment: .leading, spacing: H3Spacing.small) {
          Text("Quality")
            .font(.system(size: 12, weight: .semibold))
          Picker("Quality", selection: $quality) {
            ForEach(EngineGenerationQuality.allCases, id: \.self) { tier in
              Text(tier.displayName).tag(tier)
            }
          }
          .pickerStyle(.segmented)
          .labelsHidden()
          .accessibilityIdentifier("generation-quality")
          Text(quality.guidance)
            .font(.system(size: 10))
            .foregroundStyle(H3Color.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .onChange(of: quality) { _, changed in
          denoisingSteps = Double(changed.denoisingSteps)
          activeDiTLayers = changed.activeDiTLayers
          coreReuse = 1
        }

        VStack(alignment: .leading, spacing: H3Spacing.small) {
          HStack {
            Text("Denoising passes")
              .font(.system(size: 12, weight: .semibold))
            Spacer()
            Text(
              Int(denoisingSteps) == quality.denoisingSteps
                ? "\(Int(denoisingSteps)) · preset default"
                : "\(Int(denoisingSteps))"
            )
            .font(.system(size: 11, weight: .medium, design: .monospaced))
            .foregroundStyle(H3Color.textSecondary)
          }
          Slider(value: $denoisingSteps, in: 2...30, step: 1)
            .tint(H3Color.accent)
            .accessibilityIdentifier("generation-denoising-passes")
          Text(
            "Each pass runs the full transformer once, so generation time grows "
              + "linearly. More passes improve prompt adherence and detail; "
              + "4–7 is the validated preview band."
          )
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .fixedSize(horizontal: false, vertical: true)
        }

        VStack(alignment: .leading, spacing: H3Spacing.small) {
          HStack {
            Text("Transformer blocks")
              .font(.system(size: 12, weight: .semibold))
            Spacer()
            Picker("Transformer blocks", selection: $activeDiTLayers) {
              Text("All 50").tag(50)
              Text("Fast 45").tag(45)
              Text("Aggressive 40").tag(40)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .accessibilityIdentifier("generation-dit-layers")
          }
          HStack {
            Text("Core reuse")
              .font(.system(size: 12, weight: .semibold))
            Spacer()
            Picker("Core reuse", selection: $coreReuse) {
              Text("Off").tag(1)
              Text("Every 4th").tag(4)
              Text("Every 6th").tag(6)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .accessibilityIdentifier("generation-core-reuse")
          }
          Text(
            coreReuse > 1
              ? "Fewer blocks trim ~10–20% per pass. Core reuse runs the "
                + "expensive core only every Nth pass — worthwhile from about "
                + "10 passes up, and it disables whole-denoiser reuse."
              : "Fewer blocks trim ~10–20% per pass with a small quality cost. "
                + "Fast 45 is the validated setting; Aggressive 40 can drift."
          )
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .fixedSize(horizontal: false, vertical: true)
        }

        Toggle(isOn: $model.previewDenoise) {
          VStack(alignment: .leading, spacing: 4) {
            Text("Denoising preview")
              .font(.system(size: 12, weight: .semibold))
            Text(
              "Decode a still after every pass so you can cancel early. "
                + "Each still is a full VideoVAE pass and does not change the final video."
            )
            .font(.system(size: 10))
            .foregroundStyle(H3Color.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
          }
        }
        .toggleStyle(.switch)
        .tint(H3Color.accent)
        .accessibilityIdentifier("generation-preview-toggle")
      }

      if model.isGenerating {
        VStack(alignment: .leading, spacing: 7) {
          if let preview = model.generationPreviewImage {
            VStack(alignment: .leading, spacing: H3Spacing.small) {
              Text("Denoising preview")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(H3Color.textSecondary)
              Image(decorative: preview, scale: 1)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: .infinity)
                .frame(maxHeight: 220)
                .clipShape(
                  RoundedRectangle(cornerRadius: H3Radius.medium, style: .continuous)
                )
                .overlay {
                  RoundedRectangle(cornerRadius: H3Radius.medium, style: .continuous)
                    .stroke(H3Color.line, lineWidth: 1)
                }
                .accessibilityIdentifier("generation-denoise-preview")
            }
          }
          HStack {
            Text(model.generationPhase)
            Spacer()
            Text(
              "\(model.generationElapsedDescription) · \(Int(model.generationProgress * 100))%"
            )
            .accessibilityIdentifier("generation-elapsed")
          }
          .font(.system(size: 11, weight: .medium))
          .foregroundStyle(H3Color.textSecondary)
          ProgressView(value: model.generationProgress)
            .tint(H3Color.accent)
        }
        .transition(.opacity.combined(with: .move(edge: .bottom)))
      }

      if let errorMessage = model.errorMessage {
        Text(errorMessage)
          .font(.system(size: 11, weight: .medium))
          .foregroundStyle(H3Color.danger)
      }
    }
    .padding(H3Spacing.large)
    .frame(maxWidth: .infinity, alignment: .topLeading)
    .animation(.easeOut(duration: 0.18), value: model.isGenerating)
  }

  private var footer: some View {
    HStack {
      Text(
        model.generationBackendDescription(
          for: kind,
          quality: quality,
          denoisingSteps: Int(denoisingSteps),
          activeDiTLayers: activeDiTLayers,
          coreReuse: coreReuse,
          previewDenoise: model.previewDenoise
        )
      )
        .font(.system(size: 11))
        .foregroundStyle(H3Color.textSecondary)
        .lineLimit(2)
        .fixedSize(horizontal: false, vertical: true)
      Spacer()
      if model.isGenerating {
        Button("Cancel") {
          model.cancelGeneration()
        }
        .buttonStyle(H3QuietButtonStyle())
      }
      Button("Generate & append") {
        model.generate(
          prompt: model.generationPrompt,
          duration: requestedDuration,
          quality: quality,
          denoisingSteps: Int(denoisingSteps),
          activeDiTLayers: activeDiTLayers,
          coreReuse: coreReuse,
          previewDenoise: model.previewDenoise
        )
      }
      .buttonStyle(H3PrimaryButtonStyle())
      .disabled(
        model.generationPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
          || model.isGenerating
      )
      .opacity(
        model.generationPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.5 : 1
      )
      .accessibilityIdentifier("generate-and-append")
    }
    .padding(.horizontal, H3Spacing.large)
    .padding(.vertical, H3Spacing.medium)
    .frame(minHeight: 62)
  }

  private var alignedFrames: Int {
    Self.h3MinimumFrames + Self.h3FrameChunk * Int(alignedDurationStep)
  }

  private var alignedSeconds: Double {
    Double(alignedFrames) / Self.h3FPS
  }

  private var usesNativeVisualSettings: Bool {
    (kind == .video && model.nativeVideoGenerationIsReady)
      || (kind == .image && model.nativeImageGenerationIsReady)
  }

  private var requestedDuration: Double {
    guard kind == .video, model.nativeVideoGenerationIsReady else { return duration }
    // The engine rounds seconds up to the next legal frame shape; asking for
    // half a frame less than the target keeps float error from tipping the
    // request into the next 17-frame chunk.
    return (Double(alignedFrames) - 0.5) / Self.h3FPS
  }
}

extension EngineGenerationQuality {
  fileprivate var displayName: String {
    switch self {
    case .preview: "Preview"
    case .standard: "Standard"
    case .high: "High"
    }
  }

  fileprivate var guidance: String {
    let canvas = "\(canvasSize)×\(canvasSize)"
    return switch self {
    case .preview:
      "\(canvas) · \(denoisingSteps) passes · fastest iteration; the fit for M1/M2-class Macs."
    case .standard:
      "\(canvas) · \(denoisingSteps) passes · validated fast settings; comfortable from M3-class Macs."
    case .high:
      "\(canvas) · \(denoisingSteps) passes · close quality; intended for M5-class Macs."
    }
  }
}
