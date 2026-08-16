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
  @State private var summaryCopied = false

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
        summaryCopied = false
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
          // Nothing to describe until something can generate it.
          if hasUsableModel {
            promptPane
              .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider().overlay(H3Color.line.opacity(0.75))
          }
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
            choices: model.installedModelChoices(for: kind),
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
      if kind == .audio {
        // Two different models, not two settings of one: H3 writes the
        // joint soundtrack and leans toward speech; music and sound effects
        // are one Stable Audio transformer trained on different material.
        H3SegmentedControl(
          selection: $model.audioMode,
          segments: [
            .init(value: .voice, title: "VOICE", systemImage: "mic"),
            .init(value: .music, title: "MUSIC", systemImage: "music.note"),
            .init(value: .soundEffects, title: "SFX", systemImage: "waveform"),
          ],
          isEnabled: !model.isGenerating
        )
        .padding(.leading, 4)
      }
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

  /// Whether anything installed can produce this kind of output. Without
  /// one there is nothing to describe and no length to choose, so the
  /// controls would be asking for settings that cannot be used.
  private var hasUsableModel: Bool {
    !model.installedModelChoices(for: kind).isEmpty
  }

  private var noModelSection: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("No models installed")
        .font(.system(size: 13, weight: .semibold))
      Text("Generating \(kindNoun) needs a model on this Mac. "
        + "Get one from the model library.")
        .font(.system(size: 11))
        .foregroundStyle(H3Color.textSecondary)
        .fixedSize(horizontal: false, vertical: true)
      Button("Open Models") {
        model.showsModelSettings = true
      }
      .buttonStyle(H3PrimaryButtonStyle())
      .accessibilityIdentifier("open-models-from-studio")
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private var kindNoun: String {
    switch kind {
    case .video: "video"
    case .image: "images"
    case .audio: "audio"
    }
  }

  private var promptPane: some View {
    VStack(alignment: .leading, spacing: 18) {
      Text("PROMPT")
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .tracking(1.6)
        .foregroundStyle(H3Color.textSecondary.opacity(0.75))

      ZStack(alignment: .bottomLeading) {
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

        if kind != .audio, let query = mentionQuery {
          mentionPicker(query: query)
            .padding(10)
        }
      }

      if kind != .image, model.audioMode == .voice {
        audioDesignSection
      }

      if kind != .audio {
        frameAnchorSection
        referenceSection
        if let note = conditioningNote {
          Text(note)
            .font(.system(size: 10))
            .foregroundStyle(H3Color.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
    }
    .padding(20)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    .background(H3Color.surface)
  }

  /// The soundscape and music sections of H3's trained prompt schema. The
  /// model reads ambience from a dedicated field, not from the main prose —
  /// a rain prompt without one has been observed coming back as speech.
  private var audioDesignSection: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text("AUDIO DESIGN")
          .font(.system(size: 9, weight: .bold, design: .monospaced))
          .tracking(1.6)
          .foregroundStyle(H3Color.textSecondary.opacity(0.75))
        Spacer()
        Text("optional")
          .font(.system(size: 9, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary.opacity(0.55))
      }
      TextField(
        "Soundscape — ambient and action sound, e.g. steady rain on leaves",
        text: $model.studioSoundscape,
        axis: .vertical
      )
      .textFieldStyle(.plain)
      .font(.system(size: 12))
      .lineLimit(1...3)
      .padding(10)
      .background(H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: 9, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
      .accessibilityIdentifier("generation-soundscape")
      TextField(
        "Music — background score the characters cannot hear; empty for none",
        text: $model.studioMusic,
        axis: .vertical
      )
      .textFieldStyle(.plain)
      .font(.system(size: 12))
      .lineLimit(1...3)
      .padding(10)
      .background(H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: 9, style: .continuous)
          .stroke(H3Color.line, lineWidth: 1)
      }
      .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
      .accessibilityIdentifier("generation-music")
    }
  }

  private var frameAnchorSection: some View {
    let disabled = model.studioHasReferences
    return VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text("START / END FRAME")
          .font(.system(size: 9, weight: .bold, design: .monospaced))
          .tracking(1.6)
          .foregroundStyle(H3Color.textSecondary.opacity(0.75))
        Spacer()
        Text("optional")
          .font(.system(size: 9, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary.opacity(0.55))
      }
      HStack(spacing: 10) {
        frameWell(
          title: "Start",
          attachment: model.studioStartFrame,
          set: { model.setStudioStartFrame($0) },
          clear: { model.clearStudioStartFrame() }
        )
        frameWell(
          title: "End",
          attachment: model.studioEndFrame,
          set: { model.setStudioEndFrame($0) },
          clear: { model.clearStudioEndFrame() }
        )
        Spacer(minLength: 0)
      }
    }
    .opacity(disabled ? 0.38 : 1)
    .allowsHitTesting(!disabled)
  }

  private var referenceSection: some View {
    let disabled = model.studioHasFrameAnchors
    return VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text("REFERENCES")
          .font(.system(size: 9, weight: .bold, design: .monospaced))
          .tracking(1.6)
          .foregroundStyle(H3Color.textSecondary.opacity(0.75))
        Spacer()
        Text(
          model.studioReferenceImages.isEmpty
            ? "optional"
            : "\(model.studioReferenceImages.count)/\(AppModel.studioReferenceLimit)")
          .font(.system(size: 9, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary.opacity(0.55))
      }
      ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 10) {
          ForEach(Array(model.studioReferenceImages.enumerated()), id: \.element.id) { index, item in
            StudioImageCard(
              url: item.url,
              caption: "Picture \(index + 1)",
              onRemove: { model.removeStudioReference(item.id) }
            )
          }
          if model.studioReferenceImages.count < AppModel.studioReferenceLimit {
            addImageWell {
              pickImages(allowMultiple: true) { urls in
                urls.forEach(model.addStudioReference)
              }
            }
          }
        }
      }
    }
    .opacity(disabled ? 0.38 : 1)
    .allowsHitTesting(!disabled)
    .accessibilityIdentifier("generation-references")
  }

  private func frameWell(
    title: String,
    attachment: StudioImageAttachment?,
    set: @escaping (URL) -> Void,
    clear: @escaping () -> Void
  ) -> some View {
    Group {
      if let attachment {
        StudioImageCard(url: attachment.url, caption: title, onRemove: clear)
      } else {
        addImageWell(title: title) {
          pickImages(allowMultiple: false) { urls in
            if let url = urls.first { set(url) }
          }
        }
      }
    }
  }

  private func addImageWell(title: String? = nil, action: @escaping () -> Void) -> some View {
    Button(action: action) {
      VStack(spacing: 6) {
        ZStack {
          RoundedRectangle(cornerRadius: 10, style: .continuous)
            .strokeBorder(H3Color.line, style: StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
            .background(H3Color.chrome.clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous)))
          Image(systemName: "plus")
            .font(.system(size: 17, weight: .medium))
            .foregroundStyle(H3Color.textSecondary)
        }
        .frame(width: 78, height: 78)
        Text(title ?? "Add")
          .font(.system(size: 9.5, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
      }
    }
    .buttonStyle(.plain)
  }

  private func mentionPicker(query: String) -> some View {
    let needle = query.lowercased()
    let options = model.studioReferenceImages.enumerated().compactMap { index, item -> (String, String)? in
      let label = "Picture \(index + 1)"
      if needle.isEmpty || label.lowercased().contains(needle) || "\(index + 1)".hasPrefix(needle) {
        return (label, "<Picture \(index + 1)>")
      }
      return nil
    }
    return VStack(alignment: .leading, spacing: 0) {
      if options.isEmpty {
        Text(model.studioHasReferences ? "No matching picture" : "Add a reference to mention it")
          .font(.system(size: 11))
          .foregroundStyle(H3Color.textSecondary)
          .padding(10)
      } else {
        ForEach(options, id: \.0) { label, token in
          Button {
            insertMention(token)
          } label: {
            HStack {
              Text("@\(label)")
                .font(.system(size: 12, weight: .semibold))
              Spacer()
              Text(token)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(H3Color.textSecondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .contentShape(Rectangle())
          }
          .buttonStyle(.plain)
        }
      }
    }
    .frame(width: 240)
    .background(H3Color.surface)
    .overlay {
      RoundedRectangle(cornerRadius: 10, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    .shadow(color: .black.opacity(0.45), radius: 16, y: 8)
  }

  private var mentionQuery: String? {
    guard let match = model.generationPrompt.range(of: #"@([A-Za-z0-9 ]*)$"#, options: .regularExpression)
    else { return nil }
    return String(model.generationPrompt[match].dropFirst())
  }

  private func insertMention(_ token: String) {
    guard let match = model.generationPrompt.range(of: #"@[A-Za-z0-9 ]*$"#, options: .regularExpression)
    else { return }
    model.generationPrompt.replaceSubrange(match, with: token + " ")
  }

  private var conditioningNote: String? {
    // Start/end frames ride the FL2VA transformer every package ships, so the
    // optimized single-file build handles them; only ordered references need
    // the separate Ref2VA checkpoint, which the next clause covers.
    if model.studioHasReferences, model.modelReport?.hasReferenceTransformer == false {
      return "Ordered references need the Ref2VA transformer in the selected model folder."
    }
    if model.studioHasFrameAnchors, model.studioHasReferences {
      return "Start/end frames cannot be combined with references."
    }
    return nil
  }

  private func pickImages(allowMultiple: Bool, handle: ([URL]) -> Void) {
    let panel = NSOpenPanel()
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = allowMultiple
    panel.allowedContentTypes = [.png, .jpeg, .heic, .webP, .tiff, .bmp]
    panel.prompt = "Add"
    guard panel.runModal() == .OK else { return }
    handle(panel.urls)
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

          if hasUsableModel {
            if kind != .image {
              durationSection
            }
            modelSection
          } else {
            noModelSection
          }

          if usesNativeSettings, model.audioMode == .voice {
            generationControls
          }

          if usesNativeSettings, model.audioMode != .voice {
            soundEffectControls
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
          if usesNativeSettings, model.audioMode == .voice {
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
              coreReuse: knobs.coreReuse,
              blockCache: knobs.blockCache,
              fastStill: knobs.fastStill
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
      if kind == .audio, model.nativeAudioGenerationIsReady,
        model.audioMode == .voice
      {
        Text(
          "H3 has no audio-only model. It generates a \(AppModel.audioCanvasLabel) clip "
            + "and keeps the soundtrack, so audio costs about as much as video."
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
        choices: model.installedModelChoices(for: kind),
        selectedID: model.selectedModelID,
        isOpen: $modelMenuOpen,
        onFrameChange: { modelPickerFrame = $0 }
      )
    }
  }

  /// The one knob this model has. It is distilled to eight passes, so this
  /// is for finding out what that trade actually costs rather than a
  /// quality ladder to climb.
  private var soundEffectControls: some View {
    labeled("PASSES") {
      HStack(spacing: 10) {
        Slider(
          value: Binding(
            get: { Double(model.studioSettings.knobs.denoisingSteps) },
            set: { value in
              model.updateStudioKnobs { knobs in
                knobs.denoisingSteps = Int(value.rounded())
              }
            }
          ),
          in: 1...24,
          step: 1
        )
        .tint(H3Color.accent)
        Text("\(model.studioSettings.knobs.denoisingSteps)")
          .font(.system(size: 11, weight: .medium, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
          .frame(width: 22, alignment: .trailing)
        Button("Reset") {
          model.updateStudioKnobs { knobs in
            knobs.denoisingSteps = AppModel.soundEffectDefaultSteps
          }
        }
        .buttonStyle(H3QuietButtonStyle())
      }
      Text("Trained for 8. Fewer degrades quickly; more rarely helps.")
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
    }
    .accessibilityIdentifier("sound-effect-passes")
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
              Button(canvasMenuLabel(canvas)) {
                model.updateStudioKnobs { $0.canvas = canvas }
              }
            }
          } label: {
            HStack {
              Text(knobs.canvas.label)
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

      if kind == .image {
        labeled("STILL DETAIL") {
          settingChips(
            selection: knobs.fastStill,
            options: [(false, "Full 22f"), (true, "Fast 5f")]
          ) { fast in
            model.updateStudioKnobs { $0.fastStill = fast }
          }
          .accessibilityIdentifier("generation-still-detail")
        }
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
      .opacity(knobs.blockCache ? 0.38 : 1)
      .allowsHitTesting(!knobs.blockCache)

      Toggle(isOn: Binding(
        get: { knobs.blockCache },
        set: { enabled in model.updateStudioKnobs { $0.blockCache = enabled } }
      )) {
        VStack(alignment: .leading, spacing: 4) {
          Text("Block cache")
            .font(.system(size: 12, weight: .semibold))
          Text(
            "Replay cached transformer work on stable passes — about 40% "
              + "faster at 20 passes. The result is a different take of the "
              + "same quality, and it replaces core reuse.")
            .font(.system(size: 10))
            .foregroundStyle(H3Color.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
      .toggleStyle(.switch)
      .controlSize(.small)
      .tint(H3Color.accent)
      .accessibilityIdentifier("generation-block-cache")

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
            remaining: model.generationRemainingDescription,
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
            saveResult(result)
          } label: {
            Label("Download", systemImage: "arrow.down.circle")
          }
          .buttonStyle(H3QuietButtonStyle())
          .accessibilityIdentifier("download-result")
          if model.generationSummary(for: result.asset) != nil {
            Button {
              copySummary(for: result)
            } label: {
              Label(
                summaryCopied ? "Copied" : "Copy statistics",
                systemImage: summaryCopied ? "checkmark" : "doc.on.doc"
              )
            }
            .buttonStyle(H3QuietButtonStyle())
            .accessibilityIdentifier("copy-statistics")
          }
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

  /// Copies the generated file out of the app's working storage, where it
  /// would otherwise be pruned, to wherever the user keeps their media.
  private func saveResult(_ result: GenerationResult) {
    let panel = NSSavePanel()
    panel.nameFieldStringValue = result.asset.url.lastPathComponent
    panel.canCreateDirectories = true
    panel.isExtensionHidden = false
    guard panel.runModal() == .OK, let destination = panel.url else { return }
    do {
      if FileManager.default.fileExists(atPath: destination.path) {
        try FileManager.default.removeItem(at: destination)
      }
      try FileManager.default.copyItem(at: result.asset.url, to: destination)
    } catch {
      model.errorMessage = "Could not save the file: \(error.localizedDescription)"
    }
  }

  private func copySummary(for result: GenerationResult) {
    guard let summary = model.generationSummary(for: result.asset) else { return }
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(summary, forType: .string)
    summaryCopied = true
    Task {
      try? await Task.sleep(for: .seconds(2))
      summaryCopied = false
    }
  }

  private var menuOrigin: CGPoint {
    CGPoint(
      x: modelPickerFrame.minX - studioBodyFrame.minX,
      y: modelPickerFrame.maxY - studioBodyFrame.minY + 6
    )
  }

  private func startGeneration(
    denoisingSteps: Int, coreReuse: Int, blockCache: Bool = false,
    fastStill: Bool = false
  ) {
    modelMenuOpen = false
    resultIDAtStart = model.latestStudioResult?.id
    stage = .run
    let size = knobs.canvas.dimensions(aspect: Double(model.studioAspect.fraction))
    model.generate(
      prompt: model.generationPrompt,
      duration: requestedDuration,
      quality: knobs.canvas.engineQuality,
      denoisingSteps: denoisingSteps,
      activeDiTLayers: knobs.activeDiTLayers,
      coreReuse: coreReuse,
      blockCache: blockCache,
      fastStill: fastStill,
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

  /// Menu rows spell out what a tier becomes at the chosen aspect, so the
  /// short-edge name stays honest rather than hiding the real pixels.
  private func canvasMenuLabel(_ canvas: GenerationCanvas) -> String {
    let size = canvas.dimensions(aspect: Double(model.studioAspect.fraction))
    return "\(canvas.label)  ·  \(size.width)×\(size.height)"
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
      || (kind == .audio && model.nativeAudioGenerationIsReady
        && model.audioMode == .voice)
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

private struct StudioImageCard: View {
  var url: URL
  var caption: String
  var onRemove: () -> Void

  var body: some View {
    VStack(spacing: 6) {
      ZStack(alignment: .topTrailing) {
        Group {
          if let image = NSImage(contentsOf: url) {
            Image(nsImage: image)
              .resizable()
              .scaledToFill()
          } else {
            H3Color.chrome
          }
        }
        .frame(width: 78, height: 78)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay {
          RoundedRectangle(cornerRadius: 10, style: .continuous)
            .stroke(H3Color.line, lineWidth: 1)
        }

        Button(action: onRemove) {
          Image(systemName: "xmark")
            .font(.system(size: 8, weight: .bold))
            .foregroundStyle(Color.white)
            .frame(width: 16, height: 16)
            .background(Color.black.opacity(0.62))
            .clipShape(Circle())
        }
        .buttonStyle(.plain)
        .offset(x: 5, y: -5)
        .accessibilityLabel("Remove \(caption)")
      }
      Text(caption)
        .font(.system(size: 9.5, design: .monospaced))
        .foregroundStyle(H3Color.textSecondary)
        .lineLimit(1)
        .frame(width: 78)
    }
  }
}

private struct GenerationProgressCanvas: View {
  var verb: String
  var phase: String
  var elapsed: String
  var progress: Double
  var remaining: String?
  var preview: CGImage?

  var body: some View {
    ZStack {
      Color(red: 23 / 255, green: 26 / 255, blue: 32 / 255)
      if let preview {
        // Fit, not fill: filling reports an ideal size larger than the space
        // offered, which grew this pane until the Cancel button below it was
        // pushed out of the window. Fitting also shows the whole frame, which
        // is the point of a preview.
        Image(decorative: preview, scale: 1)
          .resizable()
          .scaledToFit()
          .frame(maxWidth: .infinity, maxHeight: .infinity)
          .opacity(0.28)
      }
      VStack(spacing: 16) {
        ScanningGauge(progress: progress, remaining: remaining)
        VStack(spacing: 3) {
          Text(phase.isEmpty ? "Generating \(verb)…" : phase)
            .font(.system(size: 13.5, weight: .semibold))
          Text(elapsed)
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

private struct ScanningGauge: View {
  var progress: Double
  var remaining: String?

  var body: some View {
    TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: false)) { context in
      // Sub-expressions are hoisted and kept uniformly Double: the release
      // archive on CI runners exceeds the type-checker budget when CGFloat
      // and Double mix freely inside this closure, while debug builds pass.
      let interval: Double = context.date.timeIntervalSinceReferenceDate
      let spin: Double = interval.truncatingRemainder(dividingBy: 30) / 30 * 360
      ZStack {
        Circle()
          .stroke(Color.white.opacity(0.12), lineWidth: 16)
        Canvas { ctx, size in
          let ticks = 40
          let centerX = Double(size.width) / 2
          let centerY = Double(size.height) / 2
          let outer = min(Double(size.width), Double(size.height)) / 2
          for index in 0..<ticks {
            let angle = (Double(index) / Double(ticks)) * .pi * 2 - .pi / 2
            let unitX = cos(angle)
            let unitY = sin(angle)
            var path = Path()
            path.move(
              to: CGPoint(
                x: centerX + unitX * (outer - 8),
                y: centerY + unitY * (outer - 8)
              )
            )
            path.addLine(
              to: CGPoint(
                x: centerX + unitX * (outer - 1),
                y: centerY + unitY * (outer - 1)
              )
            )
            ctx.stroke(path, with: .color(.white.opacity(0.35)), lineWidth: 1.5)
          }
        }
        .rotationEffect(.degrees(spin))
        Circle()
          .trim(from: 0, to: min(max(progress, 0.02), 1))
          .stroke(H3Color.accent, style: StrokeStyle(lineWidth: 16, lineCap: .butt))
          .rotationEffect(.degrees(-90))
        VStack(spacing: 2) {
          Text("\(Int((progress * 100).rounded()))%")
            .font(.system(size: 26, weight: .semibold, design: .monospaced))
          if let remaining {
            Text(remaining)
              .font(.system(size: 10, weight: .medium, design: .monospaced))
              .foregroundStyle(H3Color.textSecondary)
              .accessibilityIdentifier("generation-remaining")
          }
        }
      }
      .frame(width: 130, height: 130)
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


