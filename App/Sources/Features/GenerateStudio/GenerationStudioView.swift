import AppKit
import H3ddleCore
import H3ddleDesignSystem
import H3ddleEngineProtocol
import H3ddleGeneration
import ImageIO
import SwiftUI

private struct PendingMemoryIntensiveGeneration: Identifiable {
  let id = UUID()
  let denoisingSteps: Int
  let coreReuse: Int
  let blockCache: Bool
  let fastStill: Bool
  let warning: String
}

struct GenerationStudioView: View {
  @Bindable var model: AppModel
  let kind: GenerationKind

  private let featureFlags = GenerationStudioFeatureFlags()

  private static let h3FPS = 24.0
  /// The bottom of the range the model was trained on, not the shortest clip
  /// the grid can express. Below it H3 drifts off prompt — at 73 frames about
  /// half of seeds came back unrelated to what was asked, and the app used to
  /// start at 22.
  private static let h3MinimumFrames = H3Duration.minimumFrames
  private static let h3FrameChunk = H3Duration.chunk

  @State private var stage = StudioStage.compose
  @State private var resultIDAtStart: UUID?
  @State private var modelMenuOpen = false
  @State private var modelPickerFrame: CGRect = .zero
  @State private var studioBodyFrame: CGRect = .zero
  @State private var summaryCopied = false
  @State private var pendingMemoryIntensiveGeneration: PendingMemoryIntensiveGeneration?

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
    .alert(item: $pendingMemoryIntensiveGeneration) { pending in
      Alert(
        title: Text("This generation may freeze your Mac"),
        message: Text(pending.warning),
        primaryButton: .destructive(Text("Generate Anyway")) {
          startGeneration(
            denoisingSteps: pending.denoisingSteps,
            coreReuse: pending.coreReuse,
            blockCache: pending.blockCache,
            fastStill: pending.fastStill,
            allowsLTXMemoryOvercommit: true
          )
        },
        secondaryButton: .cancel()
      )
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
            selectedID: model.selectedModelID(for: kind)
          ) { choice in
            model.selectModel(choice.id, for: kind)
            if let steps = model.defaultDenoisingSteps(for: choice) {
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
        // Two models, not settings of one: speech is Qwen3-TTS, which says the
        // words you typed in a voice cloned from a clip; music and sound
        // effects are one Stable Audio transformer trained on different
        // material, and are not alternatives for one another.
        H3SegmentedControl(
          selection: $model.audioMode,
          segments: [
            .init(value: .speech, title: "SPEECH", systemImage: "text.bubble"),
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
          .frame(width: 44, height: 44)
          .contentShape(Rectangle())
      }
      .buttonStyle(.plain)
      .disabled(model.isGenerating)
      .help("Close Generation Studio")
      .accessibilityIdentifier("generation-close")
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
      .frame(minHeight: 220)
      .layoutPriority(1)

      // H3's trained prompt schema, composed into labelled sections it was
      // taught to read. LTX takes one prose prompt and denoises its soundtrack
      // from that — `H3StructuredPrompt.compose` is skipped for any engine
      // with its own package, so these two fields were being collected and
      // dropped on the floor.
      if kind == .video, !isLTX {
        audioDesignSection
      }
      if isLTX {
        Text("The soundtrack comes from the prompt above — this model "
          + "denoises picture and sound together, so describe what should be "
          + "heard in the same sentence as what should be seen.")
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .fixedSize(horizontal: false, vertical: true)
      }

      if kind == .audio, model.audioMode == .speech {
        voiceReferenceSection
      }

      // H3 keeps a frame out of a very short clip, so anchors and references
      // apply to its stills as much as to its video. Z-Image and LTX take a
      // prompt and nothing else — and the engine *refuses* a request carrying
      // pictures rather than dropping them, so offering these would collect
      // conditioning that turns the generation into an error.
      if acceptsConditioning {
        if !supportsVideoInpainting || !model.studioHasInpaintingInput {
          frameAnchorSection
        }
        if worksFromAPicture {
          sourceStrengthControls
        } else {
          referenceSection
        }
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

  /// Only once a picture is there. A strength slider with nothing to apply
  /// it to invites the reading that it does something to a prompt-only
  /// render, which it does not.
  @ViewBuilder private var sourceStrengthControls: some View {
    if model.studioStartFrame != nil {
      labeled("HOW MUCH TO REPAINT") {
        HStack(spacing: 10) {
          Slider(value: $model.studioSourceStrength, in: 0.05...1, step: 0.05)
            .accessibilityIdentifier("generation-source-strength")
          Text("\(Int((model.studioSourceStrength * 100).rounded()))%")
            .font(.system(size: 12, weight: .medium, design: .monospaced))
            .frame(width: 46, alignment: .trailing)
        }
        Text("Lower keeps more of the picture. It chooses where the sampler "
          + "starts, so it moves in whole steps.")
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .fixedSize(horizontal: false, vertical: true)
      }
    }
  }

  private var frameAnchorSection: some View {
    let disabled = model.studioHasReferences
    return VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text(worksFromAPicture ? "PICTURE TO WORK FROM" : "START / END FRAME")
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
        if !worksFromAPicture {
          frameWell(
            title: "End",
            attachment: model.studioEndFrame,
            set: { model.setStudioEndFrame($0) },
            clear: { model.clearStudioEndFrame() }
          )
        }
        Spacer(minLength: 0)
      }
    }
    .opacity(disabled ? 0.38 : 1)
    .allowsHitTesting(!disabled)
  }

  /// Which voice speaks. Neutral is the model unconditioned — no clip, no
  /// data shipped — and everything else is a clip the user kept, cloned from
  /// once and chosen by name thereafter.
  private var voiceReferenceSection: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text("VOICE")
          .font(.system(size: 9, weight: .bold, design: .monospaced))
          .tracking(1.6)
          .foregroundStyle(H3Color.textSecondary.opacity(0.75))
        Spacer()
        Text(model.studioVoiceReference == nil ? "required" : "cloned")
          .font(.system(size: 9, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary.opacity(0.55))
      }
      HStack(spacing: 10) {
        Menu {
          Button {
            model.selectedVoiceID = nil
          } label: {
            Label("Neutral", systemImage: model.selectedVoiceID == nil
              ? "checkmark" : "waveform")
          }
          if !model.savedVoices.isEmpty {
            Divider()
            ForEach(model.savedVoices) { voice in
              Button {
                model.selectedVoiceID = voice.id
              } label: {
                Label(
                  voice.name,
                  systemImage: model.selectedVoiceID == voice.id
                    ? "checkmark" : "person.wave.2")
              }
            }
          }
          Divider()
          Button("Add from a clip…") { pickVoiceReference() }
          if let selected = model.selectedVoiceID {
            Button("Remove \(model.selectedVoiceName)", role: .destructive) {
              model.removeVoice(selected)
            }
          }
        } label: {
          HStack(spacing: 8) {
            Image(systemName: model.selectedVoiceID == nil
              ? "waveform" : "person.wave.2")
              .font(.system(size: 13, weight: .medium))
            Text(model.selectedVoiceName)
              .font(.system(size: 12))
              .lineLimit(1)
              .truncationMode(.middle)
          }
        }
        .menuStyle(.borderlessButton)
        .frame(height: 38)
        .accessibilityIdentifier("speech-voice")
      }
      Text(model.selectedVoiceID == nil
        ? "Neutral is the model's own voice, with nothing to clone from. Add a "
          + "clip to speak in someone else's."
        : "A few seconds of clean speech is enough — the encoder averages over "
          + "the whole clip, so a longer one adds nothing.")
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
        .fixedSize(horizontal: false, vertical: true)
    }
    .accessibilityIdentifier("speech-voice-reference")
  }

  private func pickVoiceReference() {
    let panel = NSOpenPanel()
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    panel.allowedContentTypes = [.audio, .movie]
    panel.prompt = "Use Voice"
    guard panel.runModal() == .OK, let url = panel.urls.first else { return }
    model.addVoice(from: url)
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
            ? (supportsVideoInpainting && model.studioHasInpaintingInput
              ? "required" : "optional")
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

  /// Ref2VA inpainting takes the footage it is changing, a hard black/white
  /// mask, and at least one ordered picture describing the replacement. The
  /// source and mask stay visible as filenames because a video does not have
  /// a single honest thumbnail, and an animated mask may change every frame.
  private var videoInpaintingSection: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text("MASKED SOURCE")
          .font(.system(size: 9, weight: .bold, design: .monospaced))
          .tracking(1.6)
          .foregroundStyle(H3Color.textSecondary.opacity(0.75))
        Spacer()
        Text(model.studioHasInpaintingInput ? "inpainting" : "optional")
          .font(.system(size: 9, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary.opacity(0.55))
      }

      HStack(spacing: 10) {
        inpaintingFileWell(
          title: "Source clip",
          systemImage: "film",
          url: model.studioInpaintSourceURL,
          choose: pickInpaintSource,
          clear: model.clearStudioInpaintSource
        )
        inpaintingFileWell(
          title: model.studioInpaintMaskKind == .video ? "Mask clip" : "Mask image",
          systemImage: "circle.lefthalf.filled",
          url: model.studioInpaintMaskURL,
          choose: pickInpaintMask,
          clear: model.clearStudioInpaintMask
        )
      }

      HStack(spacing: 8) {
        Text("MASK")
          .font(.system(size: 9, weight: .bold, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary.opacity(0.7))
        settingChips(
          selection: model.studioInpaintMaskKind,
          options: [(.still, "Still"), (.video, "Video")],
          set: model.setStudioInpaintMaskKind
        )
        .frame(width: 150)
        Spacer()
        Toggle("Keep source audio", isOn: $model.studioPreservesInpaintAudio)
          .toggleStyle(.checkbox)
          .font(.system(size: 10))
      }

      Text("White is repainted; black is preserved. Use a hard black-and-white "
        + "mask. The source, video mask, and requested duration must match at 24 fps.")
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
        .fixedSize(horizontal: false, vertical: true)
    }
    .accessibilityIdentifier("generation-video-inpainting")
  }

  private func inpaintingFileWell(
    title: String,
    systemImage: String,
    url: URL?,
    choose: @escaping () -> Void,
    clear: @escaping () -> Void
  ) -> some View {
    HStack(spacing: 9) {
      Button(action: choose) {
        HStack(spacing: 8) {
          Image(systemName: systemImage)
            .font(.system(size: 14, weight: .medium))
          VStack(alignment: .leading, spacing: 2) {
            Text(title)
              .font(.system(size: 10, weight: .semibold))
            Text(url?.lastPathComponent ?? "Choose…")
              .font(.system(size: 9, design: .monospaced))
              .foregroundStyle(H3Color.textSecondary)
              .lineLimit(1)
          }
          Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .frame(maxWidth: .infinity, minHeight: 46)
        .background(H3Color.chrome)
        .overlay {
          RoundedRectangle(cornerRadius: 9, style: .continuous)
            .stroke(H3Color.line, lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
      }
      .buttonStyle(.plain)
      if url != nil {
        Button(action: clear) {
          Image(systemName: "xmark")
            .font(.system(size: 10, weight: .bold))
            .foregroundStyle(H3Color.textSecondary)
        }
        .buttonStyle(.plain)
      }
    }
    .frame(maxWidth: .infinity)
  }

  private func pickInpaintSource() {
    let panel = NSOpenPanel()
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    panel.allowedContentTypes = [.movie]
    panel.prompt = "Use Source"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    model.setStudioInpaintSource(url)
  }

  private func pickInpaintMask() {
    let panel = NSOpenPanel()
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    panel.allowedContentTypes = model.studioInpaintMaskKind == .video
      ? [.movie] : [.png, .jpeg, .heic, .webP, .tiff, .bmp]
    panel.prompt = "Use Mask"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    model.setStudioInpaintMask(url)
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

  /// Models that render from the prompt alone: no anchors, no references, and
  /// a square canvas. Both of them refuse a request carrying pictures rather
  /// than quietly dropping them, so this gates what the studio offers as well
  /// as what it sends — a control that collects conditioning the engine will
  /// reject is worse than one that is not there.
  /// LTX on the video lane. Almost nothing in H3's settings column applies to
  /// it: no retained DiT blocks, no core reuse, no block cache, no denoising
  /// preview, no beta schedule, and a canvas it chooses rather than inherits.
  /// Showing those controls is not merely untidy — every one of them is a
  /// promise the engine will not keep.
  private var isLTX: Bool { kind == .video && model.videoEngine == .ltx }

  private var supportsVideoInpainting: Bool {
    featureFlags.h3MaskedSource && kind == .video && model.videoEngine == .h3
  }

  private var promptOnlyModel: Bool {
    (kind == .image && model.imageEngine == .zImage)
      || (kind == .video && model.videoEngine == .ltx)
  }

  /// Whether this lane can be given pictures to condition on. Separate from
  /// `promptOnlyModel`, which is about the canvas and the aspect ratio: LTX
  /// renders a square it chooses *and* takes start and end frames, so the two
  /// questions have different answers and conflating them is what hid this.
  private var acceptsConditioning: Bool {
    if kind == .audio { return false }
    if kind == .image { return model.imageEngine == .h3 || worksFromAPicture }
    return model.videoEngine.acceptsReferenceInputs
  }

  /// Z-Image takes one picture and repaints it. That is a different offer
  /// from H3's conditioning — there is no end frame to pair with and no
  /// reference to rank — so the section narrows to a single well rather than
  /// showing three inputs of which two would be refused.
  private var worksFromAPicture: Bool {
    kind == .image && model.imageEngine == .zImage
  }

  private var conditioningNote: String? {
    // Start/end frames ride the FL2VA transformer every package ships, so the
    // optimized single-file build handles them; only ordered references need
    // the separate Ref2VA checkpoint, which the next clause covers.
    if model.studioHasReferences, model.modelReport?.hasReferenceTransformer == false {
      return "Ordered references need the Ref2VA transformer in the selected model folder."
    }
    if supportsVideoInpainting, model.studioHasInpaintingInput {
      if model.studioVideoInpainting == nil {
        return "Choose both a source clip and a mask to inpaint."
      }
      if model.studioReferenceImages.isEmpty {
        return "Add at least one reference image describing what should fill the white mask."
      }
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
          // Hidden rather than shown-and-ignored while a square-only model is
          // drawing: offering a ratio the renderer cannot honour is how the
          // lane came to fail with "renders square canvases only".
          // Every model here takes its shape from this now: H3 its canvas,
          // and both prompt-only models the aspect their tier is stretched to.
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
            // Speech has no duration to choose: the line decides how long it
            // takes, and the ceiling is derived from it.
            if kind != .image, !(kind == .audio && model.audioMode == .speech) {
              durationSection
            }
            modelSection
            if supportsVideoInpainting {
              videoInpaintingSection
            }
          } else {
            noModelSection
          }

          // Every knob in here describes an H3 pass — DiT layers, core reuse,
          // how many frames a still is kept from, and the presets that set
          // them together. A package with its own engine reads none of them,
          // so both lanes show this only while H3 is the model drawing.
          if hasModelSettings,
            (kind == .video && model.videoEngine == .h3)
              || (kind == .image && model.imageEngine == .h3)
          {
            generationControls
          }

          if hasModelSettings, kind == .image, model.imageEngine == .zImage {
            imageResolutionControls
            imagePassesControls
            denoisingPreviewControl
            seedControls
          }

          if hasModelSettings, isLTX {
            ltxResolutionControls
            ltxPassesControls
            seedControls
          }

          // Both are keyed on the audio mode, which every tab shares — so
          // without the kind test they follow the audio tab's setting onto
          // video and stills, where neither model is being run.
          if hasModelSettings, kind == .audio,
            model.audioMode == .music || model.audioMode == .soundEffects
          {
            soundEffectControls
          }

          if hasModelSettings, kind == .audio, model.audioMode == .speech {
            speechControls
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
          if hasModelSettings, kind != .audio {
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
          // Derived from the trained range rather than a round number: the
          // top used to be step 20, which is 464 frames, a hundred past what
          // the model has ever been asked to draw.
          in: 0...Double(supportedLength.stepCount),
          step: 1
        )
        .tint(H3Color.accent)
      } else {
        Slider(
          value: Binding(
            // Read through the floor as well as written, so the label and
            // the slider agree with what will actually be requested. The
            // stored default is three seconds, which video no longer allows.
            get: { max(model.studioSettings.duration, minimumDuration) },
            set: { model.studioSettings.duration = $0 }
          ),
          // Video keeps the trained floor even on this path. It is reached
          // when no H3 tree has validated yet, and it used to start at one
          // second — a length the model has never been asked to draw.
          in: minimumDuration...maximumDuration
        )
        .tint(H3Color.accent)
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
        selectedID: model.selectedModelID(for: kind),
        isOpen: $modelMenuOpen,
        onFrameChange: { modelPickerFrame = $0 }
      )
    }
  }

  /// Its own ladder, because the video one is a short edge to be combined
  /// with an aspect ratio and this model renders a square. Each tier carries
  /// what it costs: the top of this list is a quarter of an hour, which is
  /// worth knowing before choosing it rather than after.
  /// LTX chooses its own square, and that choice is the biggest lever on what a
  /// clip costs: the DiT is linear in tokens, and tokens are the latent cells
  /// times the latent frames. So each rung carries its estimate, the way the
  /// still tiers do — the price of a choice belongs beside it, not ten minutes
  /// later.
  private var ltxResolutionControls: some View {
    labeled("RESOLUTION") {
      Menu {
        ForEach(LTXResolution.allCases) { tier in
          Button("\(tier.label)  ·  ~\(minutesLabel(ltxMinutes(tier)))") {
            model.updateStudioKnobs { $0.ltxResolution = tier }
          }
        }
      } label: {
        HStack {
          Text(knobs.ltxResolution.label)
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
      .menuStyle(.borderlessButton)
      .accessibilityIdentifier("generation-ltx-resolution")
      Text("\(ltxFrames) frames at 24 fps · about "
        + "\(minutesLabel(ltxMinutes(knobs.ltxResolution))) for this length. "
        + "The aspect ratio above decides the shape.")
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
        .fixedSize(horizontal: false, vertical: true)
    }
  }

  /// The frame count this duration actually renders: 8k+1 and nothing between.
  private var ltxFrames: Int {
    EngineVideoOptions.frames(forSeconds: requestedDuration)
  }

  /// Roughly what a clip costs on an M1 Pro. Fitted to three measured runs
  /// rather than guessed — 65 frames at 512², 97 at 384², 17 at 384² — and it
  /// lands within 0.2% of all three.
  ///
  /// Two terms, because only two are separable from that data: a fixed cost
  /// (the Gemma tower, the connector, both decoders) and a cost linear in
  /// token-steps. The DiT is linear in rows with no intercept, which is what
  /// makes the second term a straight line rather than a curve.
  ///
  /// A per-pixel decode term was tried and dropped: fitting three unknowns to
  /// three points reproduced them exactly and handed back a *negative* cost per
  /// pixel, which is the fit absorbing noise rather than measuring anything.
  /// Two parameters over three points is the honest version.
  private func ltxMinutes(_ tier: LTXResolution) -> Double {
    let frame = tier.frame(aspect: studioAspect)
    return (GenerationDurationEstimate.ltx(
      width: frame.width,
      height: frame.height,
      frames: ltxFrames,
      denoisingSteps: knobs.denoisingSteps
    ) ?? 0) / 60
  }

  /// The project's aspect as a plain fraction, which is what both engines'
  /// tier maths takes.
  private var studioAspect: Double { Double(model.studioAspect.fraction) }

  private var h3ResolutionControls: some View {
    labeled("RESOLUTION") {
      Menu {
        ForEach(GenerationCanvas.allCases) { canvas in
          let frame = canvas.frame(aspect: studioAspect)
          Button("\(canvas.label)  ·  \(frame.width)×\(frame.height)") {
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
      .menuStyle(.borderlessButton)
      .menuIndicator(.hidden)
      let frame = knobs.canvas.frame(aspect: studioAspect)
      Text("\(frame.width) × \(frame.height) at this aspect ratio. "
        + "H3 supports 256p preview, 512p development, and its native 768p tier.")
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
        .fixedSize(horizontal: false, vertical: true)
    }
    .accessibilityIdentifier("generation-resolution")
  }

  /// The image and LTX ladders name a short edge and take their shape from the
  /// project's aspect ratio, so the exact frame is derived rather than chosen
  /// and is deliberately not shown — the number a person picks is the tier.
  /// Two of LTX's do not render the edge they name (720 and 1080 are not
  /// multiples of 32, so they render 704 and 1088), which is recorded on
  /// `LTXResolution` rather than in the menu.

  private var imageResolutionControls: some View {
    labeled("RESOLUTION") {
      Menu {
        ForEach(ImageCanvas.allCases) { canvas in
          Button("\(canvas.label)  ·  "
            + "~\(minutesLabel(imageMinutes(canvas)))") {
            model.updateStudioKnobs { $0.imageCanvas = canvas }
          }
        }
      } label: {
        HStack {
          Text(knobs.imageCanvas.label)
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
      .menuStyle(.borderlessButton)
      .menuIndicator(.hidden)
      Text("Square only in this build, so the project's aspect ratio does "
        + "not apply. About "
        + "\(minutesLabel(imageMinutes(knobs.imageCanvas))) "
        + "on an M1 Pro; the first render after launch takes longer.")
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
        .fixedSize(horizontal: false, vertical: true)
    }
    .accessibilityIdentifier("image-resolution")
  }

  private func imageMinutes(_ canvas: ImageCanvas) -> Double {
    canvas.approximateMinutes(
      aspect: studioAspect,
      denoisingSteps: knobs.denoisingSteps
    )
  }

  private func minutesLabel(_ minutes: Double) -> String {
    minutes < 1
      ? "\(Int((minutes * 60).rounded()))s"
      : String(format: "%.1f min", minutes)
  }

  /// The same one knob, for the same reason: Z-Image is distilled to eight
  /// passes too. The ceiling is lower than sound's because a still costs
  /// minutes where a sound costs seconds, so finding out that the top of the
  /// range buys nothing is an expensive discovery to offer.
  private var imagePassesControls: some View {
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
          in: 1...16,
          step: 1
        )
        .tint(H3Color.accent)
        Text("\(model.studioSettings.knobs.denoisingSteps)")
          .font(.system(size: 11, weight: .medium, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
          .frame(width: 22, alignment: .trailing)
        Button("Reset") {
          model.updateStudioKnobs { knobs in
            knobs.denoisingSteps = AppModel.imageModelDefaultSteps
          }
        }
        .buttonStyle(H3QuietButtonStyle())
      }
      Text("Distilled for 8. Fewer degrades quickly; more mostly costs time.")
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
    }
    .accessibilityIdentifier("image-passes")
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

  /// Language and temperature. Both matter more than they look: the wrong
  /// language token gives fluent speech in the wrong accent, and temperature
  /// zero is not "most accurate" but greedy, which loops — a six-word line
  /// measured at zero ran to its whole ceiling where 0.9 stopped at 2.3s.
  private var speechControls: some View {
    VStack(alignment: .leading, spacing: 28) {
      labeled("LANGUAGE") {
        Picker(
          "",
          selection: $model.studioSpeechLanguage
        ) {
          ForEach(EngineSpeechLanguage.allCases, id: \.self) { language in
            Text(language.displayName).tag(language)
          }
        }
        .labelsHidden()
        .pickerStyle(.menu)
        .tint(H3Color.accent)
        .accessibilityIdentifier("speech-language")
        Text("Names the language token, not a translation: the line is spoken "
          + "as written.")
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .fixedSize(horizontal: false, vertical: true)
      }
      labeled("VARIATION") {
        HStack(spacing: 10) {
          Slider(
            value: $model.studioSpeechTemperature,
            in: 0.1...1.4,
            step: 0.05
          )
          .tint(H3Color.accent)
          .accessibilityIdentifier("speech-temperature")
          Text(String(format: "%.2f", model.studioSpeechTemperature))
            .font(.system(size: 11, weight: .medium, design: .monospaced))
            .foregroundStyle(H3Color.textSecondary)
            .frame(width: 34, alignment: .trailing)
          Button("Reset") {
            model.studioSpeechTemperature = EngineSpeechOptions.defaultTemperature
          }
          .buttonStyle(H3QuietButtonStyle())
        }
        Text("0.9 is the reference default. Much lower reads flat and can "
          + "stall on a repeated phrase; much higher wanders.")
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
          .fixedSize(horizontal: false, vertical: true)
      }
    }
    .accessibilityIdentifier("speech-controls")
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
        h3ResolutionControls
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
            in: GenerationKnobSnapshot.h3DenoisingStepsRange,
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

      if featureFlags.advancedH3Controls {
        labeled("TRANSFORMER BLOCKS") {
          settingChips(
            selection: knobs.activeDiTLayers,
            options: [(50, "All 50"), (45, "Fast 45"), (40, "Aggressive 40")]
          ) { layers in
            model.updateStudioKnobs { $0.activeDiTLayers = layers }
          }
          .accessibilityIdentifier("generation-dit-layers")
        }
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

      if featureFlags.advancedH3Controls {
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
      }

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

      denoisingPreviewControl

      seedControls
    }
  }

  /// H3 and Z-Image use different tiny decoders, but the offer is identical:
  /// show the current trajectory after every pass and allow an early cancel.
  private var denoisingPreviewControl: some View {
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

  /// LTX's only sampler knob. H3's column has nine and this engine reads one of
  /// them, so it gets its own control rather than a filtered version of that.
  private var ltxPassesControls: some View {
    labeled("STEPS") {
      HStack(spacing: 10) {
        Slider(
          value: Binding(
            get: { Double(model.studioSettings.knobs.denoisingSteps) },
            set: { value in
              model.updateStudioKnobs { $0.denoisingSteps = Int(value.rounded()) }
            }
          ),
          in: 1...16,
          step: 1
        )
        .tint(H3Color.accent)
        Text("\(model.studioSettings.knobs.denoisingSteps)")
          .font(.system(size: 11, weight: .medium, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
          .frame(width: 22, alignment: .trailing)
        Button("Reset") {
          model.updateStudioKnobs { $0.denoisingSteps = 8 }
        }
        .buttonStyle(.plain)
        .font(.system(size: 11))
        .foregroundStyle(H3Color.textSecondary)
      }
      .accessibilityIdentifier("generation-ltx-steps")
      Text("This checkpoint is step-distilled and was released at eight. Fewer "
        + "trades detail for time; more buys very little.")
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
        .fixedSize(horizontal: false, vertical: true)
    }
  }

  /// Its own view because it belongs to every model that draws, not to H3's
  /// pass settings it happened to sit among: a seed reproduces a Z-Image
  /// render exactly as it reproduces a clip.
  private var seedControls: some View {
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
            // The run's position, not the current phase's. A phase-local
            // fraction restarts at every stage, which reads as the bar
            // finishing and starting over several times per generation.
            progress: model.generationOverallProgress,
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
    fastStill: Bool = false,
    allowsLTXMemoryOvercommit: Bool = false
  ) {
    modelMenuOpen = false
    let effectiveActiveDiTLayers = featureFlags.effectiveActiveDiTLayers(
      knobs.activeDiTLayers)
    let effectiveCoreReuse = featureFlags.effectiveCoreReuse(coreReuse)
    // A model built for stills brings its own square ladder; the video one is
    // a short edge that the project's aspect ratio widens, which is the pair
    // this renderer refuses.
    // H3's selected tier names the short edge, the project names the ratio,
    // and the native 768x1344 envelope caps the result.
    // Each prompt-only model has its own square; they are not the same knob,
    // and using the still tier for a clip is how 768 got picked for a picture
    // and silently charged to every video.
    let size: (width: Int, height: Int) =
      isLTX
      ? knobs.ltxResolution.frame(aspect: studioAspect)
      : (promptOnlyModel
        ? knobs.imageCanvas.frame(aspect: studioAspect)
        : knobs.canvas.frame(aspect: studioAspect))
    if isLTX, !allowsLTXMemoryOvercommit,
      let warning = ltxMemoryWarning(width: size.width, height: size.height)
    {
      pendingMemoryIntensiveGeneration = PendingMemoryIntensiveGeneration(
        denoisingSteps: denoisingSteps,
        coreReuse: effectiveCoreReuse,
        blockCache: blockCache,
        fastStill: fastStill,
        warning: warning
      )
      return
    }
    resultIDAtStart = model.latestStudioResult?.id
    stage = .run
    model.generate(
      prompt: model.generationPrompt,
      duration: requestedDuration,
      quality: knobs.canvas.engineQuality,
      denoisingSteps: denoisingSteps,
      activeDiTLayers: effectiveActiveDiTLayers,
      coreReuse: effectiveCoreReuse,
      blockCache: blockCache,
      fastStill: fastStill,
      previewDenoise: model.previewDenoise,
      allowsLTXMemoryOvercommit: allowsLTXMemoryOvercommit,
      seed: hasModelSettings ? model.studioSettings.seed : nil,
      canvasWidth: kind == .audio ? nil : size.width,
      canvasHeight: kind == .audio ? nil : size.height
    )
  }

  private func ltxMemoryWarning(width: Int, height: Int) -> String? {
    let conditioningPictures =
      (model.studioStartFrame == nil ? 0 : 1)
      + (model.studioEndFrame == nil ? 0 : 1)
      + model.studioReferenceImages.count
    let estimatedMemory = EngineVideoOptions.estimatedLTXPeakMemoryBytes(
      width: width,
      height: height,
      frames: ltxFrames,
      conditioningPictures: conditioningPictures
    )
    let safeMemory = EngineVideoOptions.safeLTXMemoryBudget(
      physicalMemory: ProcessInfo.processInfo.physicalMemory)
    guard estimatedMemory > safeMemory else { return nil }

    let gibibyte = 1_073_741_824.0
    let estimate = Double(estimatedMemory) / gibibyte
    let budget = Double(safeMemory) / gibibyte
    let pictures = conditioningPictures == 1 ? "picture" : "pictures"
    return String(
      format: "This %d×%d, %d-frame request with %d conditioning %@ is "
        + "estimated to need %.1f GiB of unified memory, above this Mac's "
        + "%.1f GiB safe budget. macOS may become unresponsive, generation "
        + "may take much longer, or the engine may fail. Close other apps "
        + "before continuing.",
      width, height, ltxFrames, conditioningPictures, pictures, estimate, budget
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
    let inpaintingIsReady = !supportsVideoInpainting || !model.studioHasInpaintingInput
      || (model.studioVideoInpainting != nil && !model.studioReferenceImages.isEmpty)
    return !model.generationPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      && !model.isGenerating && inpaintingIsReady
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

  /// Settings describe the selected installed model even while its engine is
  /// still validating. Readiness decides which provider can run; tying the
  /// controls to it made the whole section disappear on first open and only
  /// return after a model-picker change retriggered validation.
  private var hasModelSettings: Bool { hasUsableModel }

  /// Fifteen seconds is H3's ceiling, where every extra second costs a
  /// transformer pass over more frames. Stable Audio was trained to two
  /// minutes and holds about twice realtime throughout, so the same limit
  /// would be borrowing a constraint it does not have.
  ///
  /// The audio mode only decides this on the audio tab. It is one setting
  /// shared by every tab, so consulting it elsewhere hands video whichever
  /// ceiling the audio tab was last left on.
  /// What the model on this lane will actually generate. Asked of the engine
  /// rather than hardcoded per tab, so a lane cannot offer a length its model
  /// has never been trained to produce.
  private var supportedLength: SupportedLength {
    switch kind {
    // Which model is chosen decides this, not which tab: LTX's grid is 8k+1
    // frames from 17, where H3's is 17k+5 from 124. Offering one model the
    // other's range is offering lengths it cannot draw.
    case .video: model.videoEngine == .ltx ? .ltxVideo : .h3Video
    case .image: .still
    case .audio:
      switch model.audioMode {
      // A ceiling rather than a length: speech stops when the line is spoken,
      // so this only decides how long a runaway is allowed to run.
      case .speech: .speechCeiling
      case .music, .soundEffects: .stableAudio
      }
    }
  }

  private var minimumDuration: Double { supportedLength.minimumSeconds }

  private var maximumDuration: Double {
    supportedLength.maximumSeconds
  }

  private var usesAlignedH3Duration: Bool {
    kind == .video && model.videoEngine == .h3 && model.nativeVideoGenerationIsReady
  }

  private var requestedDuration: Double {
    if kind == .image { return 3 }
    if kind == .audio, model.audioMode == .speech { return speechCeiling }
    guard usesAlignedH3Duration else {
      // Resolved rather than trusted: the slider is bounded, a restored
      // project's stored duration is not, and neither is a length typed by
      // any other caller.
      return supportedLength.resolved(model.studioSettings.duration)
    }
    return (Double(alignedFrames) - 0.5) / Self.h3FPS
  }

  /// How long the line is allowed to take, from how long the line is.
  ///
  /// A fixed ceiling is wrong in both directions: too low truncates a long
  /// line mid-word, and too high leaves the progress bar reporting a fraction
  /// of a length that was never going to happen — the engine counts frames
  /// against this, so a thirty-second ceiling on a five-second line shows 17%
  /// and then jumps to done. English runs about fourteen characters a second,
  /// so ten gives roughly half again as much room as the line should need.
  private var speechCeiling: Double {
    let characters = model.generationPrompt
      .trimmingCharacters(in: .whitespacesAndNewlines).count
    return min(maximumDuration, max(8, Double(characters) / 10))
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

  @State private var thumbnail: CGImage?

  var body: some View {
    VStack(spacing: 6) {
      ZStack(alignment: .topTrailing) {
        Group {
          if let thumbnail {
            Image(decorative: thumbnail, scale: 1)
              .resizable()
              .interpolation(.high)
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
    .task(id: url) {
      thumbnail = nil
      let loaded = await Task.detached(priority: .utility) {
        Self.loadThumbnail(at: url)
      }.value
      guard !Task.isCancelled else { return }
      thumbnail = loaded
    }
  }

  nonisolated private static func loadThumbnail(at url: URL) -> CGImage? {
    autoreleasepool {
      let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
      guard let source = CGImageSourceCreateWithURL(url as CFURL, sourceOptions) else {
        return nil
      }
      let thumbnailOptions: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: 156,
        kCGImageSourceShouldCacheImmediately: true,
      ]
      return CGImageSourceCreateThumbnailAtIndex(source, 0, thumbnailOptions as CFDictionary)
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
      LocalImagePreview(url: asset.url) {
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
