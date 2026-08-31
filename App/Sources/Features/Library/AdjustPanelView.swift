import AVKit
import H3ddleCore
import H3ddleDesignSystem
import SwiftUI

struct AdjustPanelView: View {
  @Bindable var model: AppModel
  @State private var nameDraft = ""
  @State private var transformOpen = true
  @State private var mediaSize: CanvasMediaDimensions?

  private var selectedAsset: AssetReference? {
    if let id = model.selectedLibraryAssetID {
      return model.project.asset(id: id)
    }
    switch model.selectedTimelineItem {
    case .visual(let id):
      return model.project.timeline.visualItems.first { $0.id == id }
        .flatMap { model.project.asset(id: $0.assetID) }
    case .audio(let id):
      return model.project.timeline.audioItems.first { $0.id == id }
        .flatMap { model.project.asset(id: $0.assetID) }
    case .text, nil:
      return nil
    }
  }

  private var selectedVisual: VisualItem? {
    guard case .visual(let id) = model.selectedTimelineItem else { return nil }
    return model.project.timeline.visualItems.first { $0.id == id }
  }

  private var selectedText: TextItem? {
    guard case .text(let id) = model.selectedTimelineItem else { return nil }
    return model.project.timeline.textItems.first { $0.id == id }
  }

  private var selectedAudio: AudioItem? {
    guard case .audio(let id) = model.selectedTimelineItem else { return nil }
    return model.project.timeline.audioItems.first { $0.id == id }
  }

  private var resolutionLabel: String? {
    guard let mediaSize, mediaSize.width > 1, mediaSize.height > 1 else { return nil }
    return "\(Int(mediaSize.width.rounded()))×\(Int(mediaSize.height.rounded()))"
  }

  private var header: Header? {
    if let visual = selectedVisual, let asset = selectedAsset {
      return Header(
        name: asset.displayName,
        typeLabel: typeLabel(asset.kind),
        duration: visual.duration,
        track: "V1",
        tag: H3Color.clipVideo,
        resolution: resolutionLabel
      )
    }
    if let audio = selectedAudio, let asset = selectedAsset {
      return Header(
        name: asset.displayName,
        typeLabel: typeLabel(asset.kind),
        duration: audio.duration,
        track: "A1",
        tag: H3Color.clipAudio
      )
    }
    if let text = selectedText {
      let title = text.text.split(whereSeparator: \.isNewline).first.map(String.init) ?? text.text
      return Header(
        name: title.isEmpty ? "Text" : title,
        typeLabel: "Title",
        duration: text.duration,
        track: "T1",
        tag: H3Color.clipText
      )
    }
    if let asset = selectedAsset {
      return Header(
        name: asset.displayName,
        typeLabel: typeLabel(asset.kind),
        duration: asset.duration,
        track: "Library",
        tag: tag(for: asset.kind),
        resolution: asset.kind.isVisual ? resolutionLabel : nil
      )
    }
    return nil
  }

  var body: some View {
    Group {
      if let header {
        ScrollView {
          VStack(alignment: .leading, spacing: 0) {
            headerRow(header)
            preview
              .padding(.bottom, 14)
            if let asset = selectedAsset, asset.kind.isVisual {
              visualActionRow
                .padding(.bottom, 16)
            }
            if let visual = selectedVisual {
              framingSection(visual)
              transformSection(
                transform: visual.canvasTransform,
                onChange: { model.setVisualCanvasTransform(visual.id, $0) }
              )
            } else if let text = selectedText {
              transformSection(
                transform: text.canvasTransform,
                onChange: { model.setTextTransform(text.id, $0) }
              )
            } else {
              Text("Add this asset to the timeline to edit placement.")
                .font(.system(size: 11))
                .foregroundStyle(H3Color.textSecondary)
                .padding(.top, 4)
            }
          }
          .padding(14)
        }
        .onAppear { nameDraft = header.name }
        .onChange(of: header.name) { _, name in
          nameDraft = name
        }
        .task(id: selectedAsset?.id) {
          mediaSize = nil
          guard let asset = selectedAsset, asset.kind.isVisual else { return }
          mediaSize = await CanvasMediaSize.shared.size(for: asset)
        }
      } else {
        EmptyPanelPlaceholder(
          title: "No clip selected",
          detail: "Select a clip on the timeline to edit its properties."
        ) {
          EmptyPanelGlyph(systemName: "slider.horizontal.3")
        }
      }
    }
    .accessibilityIdentifier("adjust-panel")
  }

  private func headerRow(_ header: Header) -> some View {
    HStack(alignment: .center, spacing: 11) {
      RoundedRectangle(cornerRadius: 2, style: .continuous)
        .fill(header.tag)
        .frame(width: 4, height: 40)
      VStack(alignment: .leading, spacing: 2) {
        TextField("Name", text: $nameDraft)
          .textFieldStyle(.plain)
          .font(.system(size: 14, weight: .semibold))
          .lineLimit(1)
          .onSubmit { commitName() }
        Text(Self.metaLine(header))
          .font(.system(size: 9, weight: .medium, design: .monospaced))
          .tracking(0.8)
          .foregroundStyle(H3Color.textSecondary)
      }
      .frame(maxWidth: .infinity, alignment: .leading)
    }
    .padding(.bottom, 13)
    .overlay(alignment: .bottom) {
      Rectangle().fill(H3Color.line).frame(height: 1)
    }
    .padding(.bottom, 15)
    .onDisappear { commitName() }
  }

  @ViewBuilder
  private var preview: some View {
    ZStack {
      H3Color.chrome
      Color.black.opacity(0.2)
      if let asset = selectedAsset {
        switch asset.kind {
        case .image:
          LocalImagePreview(url: asset.url) {
            Image(systemName: "photo")
              .font(.system(size: 24))
              .foregroundStyle(H3Color.textSecondary.opacity(0.7))
          }
          .scaledToFit()
        case .video:
          MutedVideoPoster(url: asset.url)
        case .audio:
          Image(systemName: "waveform")
            .font(.system(size: 24, weight: .medium))
            .foregroundStyle(H3Color.clipAudio)
        }
      } else if let text = selectedText {
        Text(text.text)
          .font(.system(size: 18, weight: .semibold))
          .foregroundStyle(H3Color.textPrimary)
          .multilineTextAlignment(.center)
          .padding(16)
      }
    }
    .frame(maxWidth: .infinity)
    .aspectRatio(16 / 10, contentMode: .fit)
    .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
    .overlay {
      RoundedRectangle(cornerRadius: 7, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
    .accessibilityIdentifier("inspector-asset-preview")
  }

  private func framingSection(_ visual: VisualItem) -> some View {
    VStack(alignment: .leading, spacing: 7) {
      Text("Canvas framing")
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(H3Color.textSecondary)
      HStack(spacing: 6) {
        fitChip("Fit", selected: visual.canvasFit == .fit) {
          model.setVisualCanvasFit(visual.id, .fit)
        }
        fitChip("Cover", selected: visual.canvasFit == .cover) {
          model.setVisualCanvasFit(visual.id, .cover)
        }
      }
    }
    .padding(.bottom, 14)
  }

  private var visualActionRow: some View {
    HStack(spacing: 6) {
      Button {
        model.openRail(.upscale)
      } label: {
        Label("Upscale", systemImage: "arrow.up.left.and.arrow.down.right")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(H3QuietButtonStyle())
      .help("Open upscale settings")
      .accessibilityIdentifier("open-upscale-panel")

      Button {
        guard let visual = selectedVisual else { return }
        model.presentRegeneration(forVisualClip: visual.id)
      } label: {
        Label("Regenerate", systemImage: "arrow.clockwise")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(H3QuietButtonStyle())
      .disabled(!canRegenerateSelectedVisual)
      .opacity(canRegenerateSelectedVisual ? 1 : 0.4)
      .help(regenerateHelp)
      .accessibilityIdentifier("regenerate-selected-clip")
    }
  }

  private var canRegenerateSelectedVisual: Bool {
    selectedVisual.map { model.canRegenerateVisual($0.id) } ?? false
  }

  private var regenerateHelp: String {
    guard let visual = selectedVisual else {
      return "Regeneration is available for clips on the timeline"
    }
    if model.isRegeneratingVisual(visual.id) {
      return "A replacement is already being generated"
    }
    if !model.canRegenerateVisual(visual.id) {
      return "Regeneration requires saved generation settings"
    }
    return "Open this clip's generation settings"
  }

  private func fitChip(_ title: String, selected: Bool, action: @escaping () -> Void) -> some View {
    Button(action: action) {
      Text(title)
        .font(.system(size: 11, weight: .semibold))
        .frame(maxWidth: .infinity)
        .frame(height: 28)
        .background(selected ? H3Color.accent : H3Color.chrome)
        .foregroundStyle(selected ? Color.white : H3Color.textPrimary)
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }
    .buttonStyle(.plain)
  }

  private func transformSection(
    transform: CanvasObjectTransform,
    onChange: @escaping (CanvasObjectTransform) -> Void
  ) -> some View {
    let width = Double(model.project.settings.width)
    let height = Double(model.project.settings.height)
    return VStack(alignment: .leading, spacing: 0) {
      H3Accordion("Transform", isExpanded: $transformOpen) {
        VStack(alignment: .leading, spacing: 13) {
          sliderField(
            label: "Scale",
            valueText: "\(Int((transform.scale * 100).rounded()))%",
            value: transform.scale * 100,
            range: 10...400
          ) { percent in
            var next = transform
            next.scale = percent / 100
            onChange(next)
          }
          sliderField(
            label: "Position X",
            valueText: "\(Int((transform.translationX * width).rounded()))px",
            value: transform.translationX * width,
            range: -width...width
          ) { pixels in
            var next = transform
            next.translationX = width == 0 ? 0 : pixels / width
            onChange(next)
          }
          sliderField(
            label: "Position Y",
            valueText: "\(Int((transform.translationY * height).rounded()))px",
            value: transform.translationY * height,
            range: -height...height
          ) { pixels in
            var next = transform
            next.translationY = height == 0 ? 0 : pixels / height
            onChange(next)
          }
          sliderField(
            label: "Rotation",
            valueText: "\(Int(transform.rotationDegrees.rounded()))°",
            value: transform.rotationDegrees,
            range: -180...180
          ) { degrees in
            var next = transform
            next.rotationRadians = degrees * .pi / 180
            onChange(next)
          }
        }
        .padding(.top, 11)
      }
    }
  }

  private func sliderField(
    label: String,
    valueText: String,
    value: Double,
    range: ClosedRange<Double>,
    onChange: @escaping (Double) -> Void
  ) -> some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack {
        Text(label)
          .font(.system(size: 11))
          .foregroundStyle(H3Color.textPrimary.opacity(0.78))
        Spacer()
        Text(valueText)
          .font(.system(size: 10, weight: .medium, design: .monospaced))
          .foregroundStyle(H3Color.accent)
      }
      Slider(
        value: Binding(
          get: { value },
          set: onChange
        ),
        in: range
      )
      .tint(H3Color.accent)
    }
  }

  private func commitName() {
    let trimmed = nameDraft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }
    if let asset = selectedAsset {
      model.renameLibraryAsset(asset.id, to: trimmed)
    } else if let text = selectedText {
      model.setTextContent(text.id, trimmed)
    }
  }

  private func typeLabel(_ kind: MediaKind) -> String {
    switch kind {
    case .video: "Video"
    case .image: "Image"
    case .audio: "Audio"
    }
  }

  private func tag(for kind: MediaKind) -> Color {
    switch kind {
    case .video, .image: H3Color.clipVideo
    case .audio: H3Color.clipAudio
    }
  }

  private static func timecode(_ duration: TimeInterval) -> String {
    let total = max(0, Int(duration.rounded()))
    return String(format: "%d:%02d", total / 60, total % 60)
  }

  private static func metaLine(_ header: Header) -> String {
    var parts = [header.typeLabel, timecode(header.duration), header.track]
    if let resolution = header.resolution {
      parts.append(resolution)
    }
    return parts.joined(separator: "  ·  ")
  }

  private struct Header: Equatable {
    var name: String
    var typeLabel: String
    var duration: TimeInterval
    var track: String
    var tag: Color
    var resolution: String? = nil
  }
}

private struct MutedVideoPoster: NSViewRepresentable {
  let url: URL

  func makeNSView(context: Context) -> AVPlayerView {
    let view = AVPlayerView()
    view.controlsStyle = .none
    view.videoGravity = .resizeAspect
    let player = AVPlayer(url: url)
    player.isMuted = true
    view.player = player
    return view
  }

  func updateNSView(_ view: AVPlayerView, context: Context) {
    let current = (view.player?.currentItem?.asset as? AVURLAsset)?.url
    guard current != url else { return }
    view.player?.pause()
    let player = AVPlayer(url: url)
    player.isMuted = true
    view.player = player
  }

  static func dismantleNSView(_ view: AVPlayerView, coordinator: ()) {
    view.player?.pause()
    view.player = nil
  }
}
