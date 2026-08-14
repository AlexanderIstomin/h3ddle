import AppKit
import H3ddleCore
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

  @State private var stage = StudioStage.compose
  @State private var resultIDAtStart: UUID?
  @State private var modelMenuOpen = false
  @State private var modelPickerFrame: CGRect = .zero
  @State private var studioBodyFrame: CGRect = .zero

  var body: some View {
    ZStack {
      Color.black.opacity(0.62)
        .ignoresSafeArea()
        .onTapGesture {
          if !model.isGenerating {
            model.activeGenerationKind = nil
          }
        }

      VStack(spacing: 0) {
        header
        Divider().overlay(H3Color.line.opacity(0.75))
        studioBody
      }
      .frame(maxWidth: 1_080, maxHeight: 720)
      .background(H3Color.surface)
      .clipShape(RoundedRectangle(cornerRadius: H3Radius.large, style: .continuous))
      .overlay {
        RoundedRectangle(cornerRadius: H3Radius.large, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .shadow(color: .black.opacity(0.55), radius: 40, y: 18)
      .padding(24)
    }
    .foregroundStyle(H3Color.textPrimary)
    .onChange(of: model.isGenerating) { _, generating in
      guard !generating, stage == .run else { return }
      if model.errorMessage == nil,
        let latest = model.latestStudioResult,
        latest.id != resultIDAtStart
      {
        stage = .result
      } else {
        stage = .compose
      }
    }
  }

  @ViewBuilder
  private var studioBody: some View {
    switch stage {
    case .compose:
      ZStack(alignment: .topLeading) {
        HStack(spacing: 0) {
          promptPane
            .frame(maxWidth: .infinity, maxHeight: .infinity)
          Divider().overlay(H3Color.line.opacity(0.75))
          settingsPane
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }

        if modelMenuOpen {
          Color.clear
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .contentShape(Rectangle())
            .onTapGesture {
              modelMenuOpen = false
            }
            .accessibilityIdentifier("generation-model-dismiss")

          ModelDropdownMenu(
            choices: model.installedModelChoices,
            selectedID: model.selectedModelID
          ) { choice in
            model.selectModel(choice.id)
            if let steps = choice.generationProfile.defaultDenoisingSteps {
              model.updateStudioKnobs { $0.denoisingSteps = steps }
            }
            modelMenuOpen = false
          }
          .frame(width: max(modelPickerFrame.width, 220), alignment: .topLeading)
          .offset(x: menuOrigin.x, y: menuOrigin.y)
        }
      }
      .onGeometryChange(for: CGRect.self) { proxy in
        proxy.frame(in: .global)
      } action: { studioBodyFrame = $0 }
    case .run, .result:
      resultPane
    }
  }

  private var header: some View {
    HStack(spacing: 16) {
      HStack(spacing: 9) {
        Image(systemName: "wand.and.stars")
          .font(.system(size: 16, weight: .semibold))
          .foregroundStyle(H3Color.accent)
        Text("GENERATION STUDIO")
          .font(.system(size: 11, weight: .bold, design: .monospaced))
          .tracking(1.6)
          .accessibilityIdentifier("generation-studio")
      }
      Text(kind.rawValue.capitalized)
        .font(.system(size: 11, weight: .medium))
        .foregroundStyle(H3Color.textSecondary)
      Spacer()
      Button {
        model.activeGenerationKind = nil
      } label: {
        Image(systemName: "xmark")
          .font(.system(size: 13, weight: .semibold))
          .foregroundStyle(H3Color.textSecondary)
          .frame(width: 34, height: 34)
          .overlay {
            RoundedRectangle(cornerRadius: 9, style: .continuous)
              .stroke(H3Color.line, lineWidth: 1)
          }
      }
      .buttonStyle(.plain)
      .disabled(model.isGenerating)
    }
    .padding(.horizontal, 18)
    .frame(height: 58)
    .background(H3Color.chrome)
  }

  private var promptPane: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("PROMPT")
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .tracking(1.6)
        .foregroundStyle(H3Color.textSecondary.opacity(0.75))

      TextEditor(text: $model.generationPrompt)
        .font(.system(size: 15))
        .scrollContentBackground(.hidden)
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(H3Color.chrome)
        .overlay {
          RoundedRectangle(cornerRadius: 13, style: .continuous)
            .stroke(H3Color.line, lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 13, style: .continuous))
        .accessibilityIdentifier("generation-prompt")
    }
    .padding(20)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    .background(H3Color.surface)
  }

  private var settingsPane: some View {
    VStack(alignment: .leading, spacing: 0) {
      ScrollView {
        VStack(alignment: .leading, spacing: 28) {
          if kind != .audio {
            labeled("ASPECT RATIO") {
              HStack(spacing: 9) {
                ForEach(ProgramAspectRatio.allCases) { ratio in
                  aspectChip(ratio)
                }
              }
            }
          }

          if kind != .image {
            durationSection
          }

          if !model.installedModelChoices.isEmpty {
            modelSection
          }

          if usesNativeSettings {
            generationControls
          }

          if let errorMessage = model.errorMessage {
            Text(errorMessage)
              .font(.system(size: 11, weight: .medium))
              .foregroundStyle(H3Color.danger)
          }
        }
        .padding(20)
      }

      VStack(alignment: .leading, spacing: 10) {
        HStack(spacing: 10) {
          if usesNativeSettings {
            Button {
              // A composition check: same seed, canvas, and model, minimum
              // passes — the full render follows the same trajectory.
              startGeneration(denoisingSteps: 3, coreReuse: 1)
            } label: {
              VStack(spacing: 1) {
                Text("Draft")
                  .font(.system(size: 13, weight: .semibold))
                Text("3 passes")
                  .font(.system(size: 9))
                  .foregroundStyle(H3Color.textSecondary)
              }
              .frame(width: 92)
              .frame(height: 50)
              .overlay {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                  .strokeBorder(H3Color.line, lineWidth: 1)
              }
              .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(!canGenerate)
            .help("Fast pre-check of composition and prompt adherence with the same seed.")
            .accessibilityIdentifier("draft-button")
          }

          Button {
            startGeneration(
              denoisingSteps: knobs.denoisingSteps,
              coreReuse: knobs.coreReuse
            )
          } label: {
            HStack(spacing: 10) {
              Image(systemName: "sparkle")
              Text(generateLabel)
            }
            .font(.system(size: 15, weight: .semibold))
            .foregroundStyle(Color.white)
            .frame(maxWidth: .infinity)
            .frame(height: 50)
            .background(H3Color.accent.opacity(canGenerate ? 1 : 0.45))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .shadow(color: H3Color.accent.opacity(canGenerate ? 0.45 : 0), radius: 14, y: 6)
          }
          .buttonStyle(.plain)
          .disabled(!canGenerate)
          .accessibilityIdentifier("generate-button")
        }
      }
      .padding(20)
    }
    .background(H3Color.surface)
  }

  private var durationSection: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text(kind == .image ? "DISPLAY DURATION" : "DURATION")
          .font(.system(size: 9, weight: .bold, design: .monospaced))
          .tracking(1.6)
          .foregroundStyle(H3Color.textSecondary.opacity(0.75))
        Spacer()
        Text(
          usesAlignedH3Duration
            ? String(format: "%.1f s · %d frames", alignedSeconds, alignedFrames)
            : String(format: "%.0f seconds", model.studioSettings.duration)
        )
        .font(.system(size: 11, weight: .medium, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
      }
      if usesAlignedH3Duration {
        Slider(
          value: Binding(
            get: { model.studioSettings.alignedDurationStep },
            set: { model.studioSettings.alignedDurationStep = $0 }
          ),
          in: 0...20,
          step: 1
        )
        .tint(H3Color.accent)
      } else {
        Slider(
          value: Binding(
            get: { model.studioSettings.duration },
            set: { model.studioSettings.duration = $0 }
          ),
          in: 1...15,
          step: 1
        )
        .tint(H3Color.accent)
      }
      if kind == .audio, model.nativeAudioGenerationIsReady {
        Text(
          "H3 has no audio-only model. It generates a 32×32 clip and keeps the soundtrack."
        )
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
      }
      if kind == .video, model.nativeVideoGenerationIsReady, alignedSeconds > 3 {
        Text("Long H3 clips increase transformer work sharply. Start short on M1-class Macs.")
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
      }
    }
  }

  private var modelSection: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("MODEL")
        .font(.system(size: 10, weight: .bold))
        .tracking(1.1)
        .foregroundStyle(H3Color.textSecondary)
      ModelDropdown(
        choices: model.installedModelChoices,
        selectedID: model.selectedModelID,
        isOpen: $modelMenuOpen,
        onFrameChange: { modelPickerFrame = $0 }
      )
    }
  }

  private var generationControls: some View {
    VStack(alignment: .leading, spacing: 28) {
      labeled("PRESETS") {
        HStack(spacing: 6) {
          ForEach(GenerationPreset.allCases) { preset in
            let selected = model.studioSettings.preset == preset
            Button {
              model.applyStudioPreset(preset)
            } label: {
              Text(preset.label)
                .font(.system(size: 11, weight: .semibold))
                .frame(maxWidth: .infinity)
                .frame(height: 30)
                .background(selected ? H3Color.accent : H3Color.chrome)
                .foregroundStyle(selected ? Color.white : H3Color.textPrimary)
                .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            }
            .buttonStyle(.plain)
          }
        }
        .accessibilityIdentifier("generation-presets")
      }

      if kind != .audio {
        labeled("RESOLUTION") {
          Menu {
            ForEach(GenerationCanvas.allCases) { canvas in
              Button(canvas.label(isPortrait: isPortraitCanvas)) {
                model.updateStudioKnobs { $0.canvas = canvas }
              }
            }
          } label: {
            HStack {
              Text(knobs.canvas.label(isPortrait: isPortraitCanvas))
                .font(.system(size: 12, weight: .medium, design: .monospaced))
              Spacer()
              Image(systemName: "chevron.up.chevron.down")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(H3Color.textSecondary)
            }
            .padding(.horizontal, 11)
            .frame(height: 36)
            .background(H3Color.chrome)
            .overlay {
              RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(H3Color.line, lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
          }
          .buttonStyle(.plain)
          .accessibilityIdentifier("generation-resolution")
        }
      }

      labeled("DENOISING PASSES") {
        HStack {
          Slider(
            value: Binding(
              get: { Double(knobs.denoisingSteps) },
              set: { steps in
                model.updateStudioKnobs { $0.denoisingSteps = Int(steps) }
              }
            ),
            in: 2...30,
            step: 1
          )
          .tint(H3Color.accent)
          .accessibilityIdentifier("generation-denoising-passes")
          Text("\(knobs.denoisingSteps)")
            .font(.system(size: 11, weight: .medium, design: .monospaced))
            .foregroundStyle(H3Color.textSecondary)
            .frame(width: 28, alignment: .trailing)
        }
        Text("Each pass runs the full transformer once. 4–7 is the validated preview band.")
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
      }

      labeled("TRANSFORMER BLOCKS") {
        settingChips(
          selection: knobs.activeDiTLayers,
          options: [(50, "All 50"), (45, "Fast 45"), (40, "Aggressive 40")]
        ) { layers in
          model.updateStudioKnobs { $0.activeDiTLayers = layers }
        }
        .accessibilityIdentifier("generation-dit-layers")
      }

      labeled("CORE REUSE") {
        settingChips(
          selection: knobs.coreReuse,
          options: [(1, "Off"), (4, "Every 4th"), (6, "Every 6th")]
        ) { reuse in
          model.updateStudioKnobs { $0.coreReuse = reuse }
        }
        .accessibilityIdentifier("generation-core-reuse")
      }

      if kind != .audio {
        Toggle(isOn: $model.previewDenoise) {
          VStack(alignment: .leading, spacing: 4) {
            Text("Denoising preview")
              .font(.system(size: 12, weight: .semibold))
            Text("Decode a still after every pass so you can cancel early.")
              .font(.system(size: 10))
              .foregroundStyle(H3Color.textSecondary)
          }
        }
        .toggleStyle(.switch)
        .tint(H3Color.accent)
        .accessibilityIdentifier("generation-preview-toggle")
      }

      labeled("SEED") {
        HStack(spacing: 10) {
          Text(String(model.studioSettings.seed))
            .font(.system(size: 13, weight: .medium, design: .monospaced))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 11)
            .frame(height: 36)
            .background(H3Color.chrome)
            .overlay {
              RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(H3Color.line, lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
          Button {
            model.studioSettings.seed = UInt64.random(in: 1..<100_000_000)
          } label: {
            Image(systemName: "dice")
              .font(.system(size: 14, weight: .semibold))
              .frame(width: 36, height: 36)
              .background(H3Color.chrome)
              .overlay {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                  .stroke(H3Color.line, lineWidth: 1)
              }
              .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
          }
          .buttonStyle(.plain)
          .help("The same seed with the same settings reproduces a generation. Click to reroll.")
          .accessibilityIdentifier("generation-seed")
        }
      }
    }
  }

  private var resultPane: some View {
    VStack(spacing: 16) {
      ZStack {
        RoundedRectangle(cornerRadius: 14, style: .continuous)
          .fill(H3Color.chrome)
          .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
              .stroke(H3Color.line, lineWidth: 1)
          }

        if stage == .run {
          GenerationProgressCanvas(
            verb: kind.rawValue,
            phase: model.generationPhase,
            elapsed: model.generationElapsedDescription,
            progress: model.generationProgress,
            preview: model.generationPreviewImage
          )
        } else if let result = model.latestStudioResult {
          GenerationResultMedia(asset: result.asset)
        }
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))

      if stage == .run {
        Button("Cancel") {
          model.cancelGeneration()
          stage = .compose
        }
        .buttonStyle(H3QuietButtonStyle())
      } else if let result = model.latestStudioResult {
        if let generatedIn = model.generationDurationDescription(for: result.asset) {
          Text("Generated in \(generatedIn)")
            .font(.system(size: 11, weight: .medium, design: .monospaced))
            .foregroundStyle(H3Color.textSecondary)
            .accessibilityIdentifier("generation-completed-duration")
        }
        HStack(spacing: 10) {
          Button("Generate another") {
            stage = .compose
          }
          .buttonStyle(H3QuietButtonStyle())
          Button {
            model.insertToTimeline(result)
          } label: {
            Label("Insert to timeline", systemImage: "plus")
          }
          .buttonStyle(H3PrimaryButtonStyle())
          .accessibilityIdentifier("insert-to-timeline")
        }
      }
    }
    .padding(20)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(Color(red: 15 / 255, green: 18 / 255, blue: 23 / 255))
  }

  private var menuOrigin: CGPoint {
    CGPoint(
      x: modelPickerFrame.minX - studioBodyFrame.minX,
      y: modelPickerFrame.maxY - studioBodyFrame.minY + 6
    )
  }

  private func startGeneration(denoisingSteps: Int, coreReuse: Int) {
    modelMenuOpen = false
    resultIDAtStart = model.latestStudioResult?.id
    stage = .run
    let size = knobs.canvas.dimensions(isPortrait: isPortraitCanvas)
    model.generate(
      prompt: model.generationPrompt,
      duration: requestedDuration,
      quality: knobs.canvas.engineQuality,
      denoisingSteps: denoisingSteps,
      activeDiTLayers: knobs.activeDiTLayers,
      coreReuse: coreReuse,
      previewDenoise: model.previewDenoise,
      seed: usesNativeSettings ? model.studioSettings.seed : nil,
      canvasWidth: kind == .audio ? nil : size.width,
      canvasHeight: kind == .audio ? nil : size.height
    )
  }

  private func settingChips<Value: Hashable>(
    selection: Value,
    options: [(Value, String)],
    set: @escaping (Value) -> Void
  ) -> some View {
    HStack(spacing: 6) {
      ForEach(options, id: \.1) { value, title in
        let selected = selection == value
        Button {
          set(value)
        } label: {
          Text(title)
            .font(.system(size: 11, weight: .semibold))
            .frame(maxWidth: .infinity)
            .frame(height: 30)
            .background(selected ? H3Color.accent : H3Color.chrome)
            .foregroundStyle(selected ? Color.white : H3Color.textPrimary)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        }
        .buttonStyle(.plain)
      }
    }
  }

  private func labeled<Content: View>(
    _ title: String,
    @ViewBuilder content: () -> Content
  ) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      Text(title)
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .tracking(1.6)
        .foregroundStyle(H3Color.textSecondary.opacity(0.75))
      content()
    }
  }

  private func aspectChip(_ ratio: ProgramAspectRatio) -> some View {
    let selected = model.studioAspect == ratio
    return Button {
      model.studioAspect = ratio
    } label: {
      VStack(spacing: 6) {
        RoundedRectangle(cornerRadius: 3, style: .continuous)
          .stroke(selected ? H3Color.accent : H3Color.line, lineWidth: selected ? 2 : 1)
          .frame(width: chipSize(ratio).width, height: chipSize(ratio).height)
        Text(ratio.rawValue)
          .font(.system(size: 10, weight: .semibold, design: .monospaced))
          .foregroundStyle(selected ? H3Color.textPrimary : H3Color.textSecondary)
      }
      .padding(.horizontal, 8)
      .padding(.vertical, 8)
      .background(selected ? H3Color.accent.opacity(0.12) : Color.clear)
      .overlay {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
          .stroke(selected ? H3Color.accent : H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
    .buttonStyle(.plain)
  }

  private func chipSize(_ ratio: ProgramAspectRatio) -> CGSize {
    let maxEdge: CGFloat = 18
    if ratio.fraction >= 1 {
      return CGSize(width: maxEdge, height: maxEdge / ratio.fraction)
    }
    return CGSize(width: maxEdge * ratio.fraction, height: maxEdge)
  }

  private var generateLabel: String {
    switch kind {
    case .video: "Generate video"
    case .image: "Generate image"
    case .audio: "Generate audio"
    }
  }

  private var canGenerate: Bool {
    !model.generationPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      && !model.isGenerating
  }

  private var knobs: GenerationKnobSnapshot {
    model.studioSettings.knobs
  }

  private var isPortraitCanvas: Bool {
    model.studioAspect.fraction < 1
  }

  private var alignedFrames: Int {
    Self.h3MinimumFrames + Self.h3FrameChunk * Int(model.studioSettings.alignedDurationStep)
  }

  private var alignedSeconds: Double {
    Double(alignedFrames) / Self.h3FPS
  }

  private var usesNativeSettings: Bool {
    model.usesNativeEngine(for: kind)
  }

  private var usesAlignedH3Duration: Bool {
    (kind == .video && model.nativeVideoGenerationIsReady)
      || (kind == .audio && model.nativeAudioGenerationIsReady)
  }

  private var requestedDuration: Double {
    if kind == .image { return 3 }
    guard usesAlignedH3Duration else { return model.studioSettings.duration }
    return (Double(alignedFrames) - 0.5) / Self.h3FPS
  }
}

private enum StudioStage {
  case compose
  case run
  case result
}

private struct GenerationProgressCanvas: View {
  var verb: String
  var phase: String
  var elapsed: String
  var progress: Double
  var preview: CGImage?

  var body: some View {
    ZStack {
      GenerationProgressBackdrop(preview: preview)
      VStack(spacing: 14) {
        GenerationProgressSpinner()
        VStack(spacing: 3) {
          Text(phase.isEmpty ? "Generating \(verb)…" : phase)
            .font(.system(size: 13.5, weight: .semibold))
          Text("\(elapsed) · \(Int(progress * 100))%")
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .tracking(0.6)
            .foregroundStyle(H3Color.textSecondary.opacity(0.8))
            .accessibilityIdentifier("generation-elapsed")
        }
      }
    }
    .clipped()
  }
}

/// Clock-driven motion so elapsed/progress updates never restart the loop.
private struct GenerationProgressBackdrop: View {
  var preview: CGImage?

  private let deep = Color(red: 23 / 255, green: 26 / 255, blue: 32 / 255)
  private let lift = Color(red: 36 / 255, green: 42 / 255, blue: 51 / 255)

  var body: some View {
    TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: false)) { context in
      let time = context.date.timeIntervalSinceReferenceDate
      let shimmer = time.truncatingRemainder(dividingBy: 2.4) / 2.4

      GeometryReader { proxy in
        let width = proxy.size.width
        let height = proxy.size.height
        ZStack {
          deep
          LinearGradient(
            stops: [
              .init(color: deep, location: 0),
              .init(color: lift, location: 0.25),
              .init(color: deep, location: 0.5),
              .init(color: lift, location: 0.75),
              .init(color: deep, location: 1),
            ],
            startPoint: .leading,
            endPoint: .trailing
          )
          .frame(width: width * 2, height: height)
          .offset(x: -width * shimmer)

          if let preview {
            Image(decorative: preview, scale: 1)
              .resizable()
              .scaledToFill()
              .frame(width: width, height: height)
              .opacity(0.35)
          }
        }
        .frame(width: width, height: height)
        .clipped()
      }
    }
  }
}

private struct GenerationProgressSpinner: View {
  var body: some View {
    TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: false)) { context in
      let spin =
        (context.date.timeIntervalSinceReferenceDate / 0.9)
        .truncatingRemainder(dividingBy: 1) * 360
      ZStack {
        Circle()
          .stroke(Color.white.opacity(0.12), lineWidth: 3)
        Circle()
          .trim(from: 0, to: 0.28)
          .stroke(H3Color.accent, style: StrokeStyle(lineWidth: 3, lineCap: .round))
          .rotationEffect(.degrees(spin))
      }
      .frame(width: 56, height: 56)
    }
  }
}

private struct GenerationResultMedia: View {
  let asset: AssetReference

  var body: some View {
    switch asset.kind {
    case .video:
      NativeVideoPlayer(url: asset.url)
    case .image:
      if let image = NSImage(contentsOf: asset.url) {
        Image(nsImage: image)
          .resizable()
          .scaledToFit()
      } else {
        placeholder("Image preview unavailable")
      }
    case .audio:
      AudioPreviewPlayer(url: asset.url, duration: asset.duration)
    }
  }

  private func placeholder(_ title: String) -> some View {
    VStack(spacing: 8) {
      Image(systemName: asset.kind == .audio ? "waveform" : "photo")
        .font(.system(size: 22, weight: .medium))
        .foregroundStyle(H3Color.accent)
      Text(title)
        .font(.system(size: 12, weight: .medium))
        .foregroundStyle(H3Color.textSecondary)
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}

/// Compact field only. The menu is hosted by the studio so it can sit below
/// the picker and dismiss when the user clicks anywhere else.
private struct ModelDropdown: View {
  var choices: [ModelChoice]
  var selectedID: String?
  @Binding var isOpen: Bool
  var onFrameChange: (CGRect) -> Void

  private var selected: ModelChoice? {
    choices.first { $0.id == selectedID } ?? choices.first
  }

  var body: some View {
    Button {
      isOpen.toggle()
    } label: {
      HStack {
        Text(selected?.displayName ?? "Choose a model")
          .font(.system(size: 13, weight: .semibold))
        Spacer()
        Image(systemName: "chevron.down")
          .font(.system(size: 12, weight: .semibold))
          .foregroundStyle(H3Color.textSecondary.opacity(0.7))
          .rotationEffect(.degrees(isOpen ? 180 : 0))
          .animation(.easeOut(duration: 0.2), value: isOpen)
      }
      .padding(.horizontal, 12)
      .frame(height: 40)
      .background(H3Color.canvas, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
      .overlay {
        RoundedRectangle(cornerRadius: 10, style: .continuous)
          .strokeBorder(H3Color.line, lineWidth: 1)
      }
      .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .accessibilityIdentifier("generation-model-picker")
    .onGeometryChange(for: CGRect.self) { proxy in
      proxy.frame(in: .global)
    } action: { onFrameChange($0) }
  }
}

private struct ModelDropdownMenu: View {
  var choices: [ModelChoice]
  var selectedID: String?
  var pick: (ModelChoice) -> Void

  var body: some View {
    VStack(spacing: 0) {
      ForEach(choices) { choice in
        let on = choice.id == selectedID
        Button {
          pick(choice)
        } label: {
          HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
              Text(choice.displayName)
                .font(.system(size: 13, weight: .semibold))
              Text(choice.subtitle)
                .font(.system(size: 11))
                .foregroundStyle(H3Color.textSecondary)
            }
            Spacer()
            Image(systemName: "checkmark")
              .font(.system(size: 13, weight: .semibold))
              .foregroundStyle(H3Color.accent)
              .opacity(on ? 1 : 0)
          }
          .padding(.vertical, 8)
          .padding(.horizontal, 10)
          .background(
            on ? H3Color.accent.opacity(0.12) : .clear,
            in: RoundedRectangle(cornerRadius: 7, style: .continuous)
          )
          .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
      }
    }
    .padding(4)
    .background(H3Color.surface, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    .overlay {
      RoundedRectangle(cornerRadius: 10, style: .continuous)
        .strokeBorder(H3Color.line, lineWidth: 1)
    }
    .shadow(color: .black.opacity(0.55), radius: 20, y: 14)
  }
}


