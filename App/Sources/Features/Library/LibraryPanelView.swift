import AppKit
import AVFoundation
import H3ddleCore
import H3ddleDesignSystem
import H3ddleMedia
import SwiftUI
import UniformTypeIdentifiers

struct LibraryPanelView: View {
  @Bindable var model: AppModel
  var kind: MediaKind
  @State private var query = ""
  @State private var sizes: [AssetID: CGSize] = [:]
  @State private var playingID: AssetID?

  private var assets: [AssetReference] {
    let base = model.libraryAssets(kind: kind)
    let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return base }
    return base.filter { $0.displayName.localizedCaseInsensitiveContains(trimmed) }
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      searchField
      ScrollViewReader { proxy in
        ScrollView {
          VStack(alignment: .leading, spacing: 8) {
            importDropzone
            if assets.isEmpty {
              empty
            } else if kind == .audio {
              audioList
            } else {
              masonry
            }
          }
          .padding(.horizontal, 12)
          .padding(.bottom, 12)
        }
        .task(id: model.selectedLibraryAssetID) {
          guard let selectedID = model.selectedLibraryAssetID,
            assets.contains(where: { $0.id == selectedID })
          else { return }
          await Task.yield()
          withAnimation(.easeOut(duration: 0.18)) {
            proxy.scrollTo(selectedID, anchor: .center)
          }
        }
      }
    }
    .onDisappear { playingID = nil }
    .onChange(of: kind) { _, _ in
      playingID = nil
      query = ""
    }
    .task(id: assets.map(\.id)) {
      await probeSizes()
    }
  }

  private var searchField: some View {
    HStack(spacing: 6) {
      Image(systemName: "magnifyingglass")
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(H3Color.textSecondary)
      TextField(searchPlaceholder, text: $query)
        .textFieldStyle(.plain)
    }
    .padding(.horizontal, 10)
    .frame(height: 32)
    .background(H3Color.chrome)
    .overlay {
      RoundedRectangle(cornerRadius: 7, style: .continuous)
        .stroke(H3Color.line, lineWidth: 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
    .padding(.horizontal, 12)
    .padding(.top, 12)
    .padding(.bottom, 8)
  }

  private var searchPlaceholder: String {
    switch kind {
    case .video: "Search videos…"
    case .image: "Search images…"
    case .audio: "Search audio…"
    }
  }

  private var masonry: some View {
    let columns = LibraryMasonry.pack(assets, sizes: sizes)
    return HStack(alignment: .top, spacing: LibraryMasonry.gap) {
      ForEach(Array(columns.enumerated()), id: \.offset) { _, column in
        LazyVStack(spacing: LibraryMasonry.gap) {
          ForEach(column) { asset in
            LibraryAssetCard(
              model: model,
              asset: asset,
              aspect: aspect(for: asset),
              isPlaying: playingID == asset.id,
              onTogglePlay: { togglePlay(asset) },
              onEnsurePlaying: { playingID = asset.id }
            )
            .id(asset.id)
          }
        }
        .frame(maxWidth: .infinity, alignment: .top)
      }
    }
    .accessibilityIdentifier("asset-masonry")
  }

  private var audioList: some View {
    LazyVStack(spacing: 8) {
      ForEach(assets) { asset in
        LibraryAudioRow(
          model: model,
          asset: asset,
          isPlaying: playingID == asset.id,
          onTogglePlay: { togglePlay(asset) },
          onEnsurePlaying: { playingID = asset.id }
        )
        .id(asset.id)
      }
    }
    .accessibilityIdentifier("asset-audio-list")
  }

  private var importDropzone: some View {
    Button {
      presentImport()
    } label: {
      VStack(spacing: 5) {
        Image(systemName: "square.and.arrow.up")
          .font(.system(size: 18, weight: .medium))
          .foregroundStyle(H3Color.textSecondary)
        Text("Import media")
          .font(.system(size: 12, weight: .semibold))
        Text("Drag & drop")
          .font(.system(size: 10))
          .foregroundStyle(H3Color.textSecondary)
      }
      .frame(maxWidth: .infinity)
      .padding(.vertical, 16)
      .background(H3Color.chrome)
      .overlay {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
          .strokeBorder(H3Color.line, style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
      }
      .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
    .buttonStyle(.plain)
    .dropDestination(for: URL.self) { urls, _ in
      Task { await model.importToLibrary(urls) }
      return true
    }
    .accessibilityIdentifier("library-import")
  }

  private var empty: some View {
    VStack(spacing: 6) {
      Text("Nothing in this bin")
        .font(.system(size: 12, weight: .semibold))
      Text("Import or generate media. It stays here until you add it to the timeline.")
        .font(.system(size: 10))
        .foregroundStyle(H3Color.textSecondary)
        .multilineTextAlignment(.center)
    }
    .frame(maxWidth: .infinity)
    .padding(.vertical, 20)
  }

  private func aspect(for asset: AssetReference) -> CGFloat {
    if let size = sizes[asset.id], size.width > 1, size.height > 1 {
      return size.width / size.height
    }
    return asset.kind == .audio ? 1 / 0.72 : 16 / 10
  }

  private func togglePlay(_ asset: AssetReference) {
    guard asset.kind == .video || asset.kind == .audio else { return }
    playingID = playingID == asset.id ? nil : asset.id
  }

  private func probeSizes() async {
    for asset in assets {
      if asset.kind == .audio { continue }
      if sizes[asset.id] != nil { continue }
      let probed = await CanvasMediaSize.shared.size(for: asset)
      sizes[asset.id] = CGSize(width: probed.width, height: probed.height)
    }
  }

  private func presentImport() {
    let panel = NSOpenPanel()
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = true
    panel.allowedContentTypes =
      kind == .audio ? MediaImport.audioContentTypes : MediaImport.visualContentTypes
    panel.prompt = "Add"
    guard panel.runModal() == .OK, !panel.urls.isEmpty else { return }
    Task { await model.importToLibrary(panel.urls) }
  }
}

struct LibraryAssetCard: View {
  @Bindable var model: AppModel
  var asset: AssetReference
  var aspect: CGFloat
  var isPlaying: Bool
  var onTogglePlay: () -> Void
  var onEnsurePlaying: () -> Void
  @State private var isHovering = false
  @State private var currentTime: TimeInterval = 0
  @State private var seekNonce: UInt = 0
  @State private var seekTime: TimeInterval = 0

  private var isSelected: Bool {
    model.selectedLibraryAssetID == asset.id
  }

  private var canPreview: Bool {
    asset.kind == .video || asset.kind == .audio
  }

  private var progress: CGFloat {
    let duration = max(asset.duration, 0.001)
    return min(max(currentTime / duration, 0), 1)
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      ZStack(alignment: .bottom) {
        thumb
        LinearGradient(
          colors: [.clear, Color.black.opacity(0.55)],
          startPoint: .center,
          endPoint: .bottom
        )
        .opacity(isHovering || isPlaying || canPreview ? 1 : 0.85)
        .allowsHitTesting(false)
        VStack {
          Spacer(minLength: 0)
          HStack(alignment: .bottom, spacing: 6) {
            if canPreview {
              playButton
            }
            Spacer(minLength: 0)
            addButton
          }
          .padding(.horizontal, 8)
          .padding(.bottom, asset.kind == .video ? 12 : 8)
        }
        if asset.kind == .video {
          LibraryProgressStrip(progress: progress, onScrub: scrub)
        }
      }
      .frame(maxWidth: .infinity)
      .aspectRatio(min(max(aspect, 0.45), 2.2), contentMode: .fit)
      .clipped()

      VStack(alignment: .leading, spacing: 4) {
        Text(asset.displayName)
          .font(.system(size: 11, weight: .semibold))
          .lineLimit(1)
        HStack(spacing: 6) {
          if asset.kind.isVisual {
            AspectRatioGlyph(ratio: aspect)
          }
          if asset.kind != .image {
            Text(Self.timecode(asset.duration))
              .font(.system(size: 10, weight: .medium, design: .monospaced))
              .foregroundStyle(H3Color.textSecondary)
          }
          Spacer(minLength: 0)
        }
      }
      .padding(.horizontal, 8)
      .padding(.top, 7)
      .padding(.bottom, 8)
    }
    .background(H3Color.chrome)
    .overlay {
      RoundedRectangle(cornerRadius: 8, style: .continuous)
        .stroke(isSelected ? H3Color.accent : H3Color.line, lineWidth: isSelected ? 1.5 : 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    .contentShape(Rectangle())
    .onHover { isHovering = $0 }
    .onTapGesture {
      model.selectedLibraryAssetID = asset.id
    }
    .draggable(asset.id.rawValue.uuidString)
    .contextMenu {
      Button("Add to timeline") { model.insertLibraryAsset(asset.id) }
      Button("Download") { AssetFileExporter.presentDownload(of: asset) }
      Button("Rename…") { model.openRail(.adjust) }
      if model.project.usageCount(of: asset.id) == 0 {
        Button("Delete", role: .destructive) {
          model.removeLibraryAsset(asset.id)
        }
      } else {
        Button("Delete and remove clips", role: .destructive) {
          model.removeLibraryAsset(asset.id, removingClips: true)
        }
      }
    }
    .accessibilityIdentifier("asset-card")
  }

  private var thumb: some View {
    ZStack {
      H3Color.controlFill
      switch asset.kind {
      case .image:
        LocalImagePreview(url: asset.url, contentMode: .fill) {
          Image(systemName: "photo")
            .foregroundStyle(H3Color.textSecondary)
        }
      case .video:
        if isPlaying || currentTime > 0.04 {
          LibraryLoopingPlayer(
            url: asset.url,
            isMuted: false,
            isPlaying: isPlaying,
            seekNonce: seekNonce,
            seekTime: seekTime,
            onTime: { currentTime = $0 }
          )
        } else {
          LibraryVideoPoster(url: asset.url)
        }
      case .audio:
        audioWaveform
      }
    }
  }

  private var audioWaveform: some View {
    HStack(alignment: .center, spacing: 3) {
      ForEach(Array(Self.waveformBars.enumerated()), id: \.offset) { _, height in
        Capsule()
          .fill(H3Color.accent.opacity(isPlaying ? 0.95 : 0.78))
          .frame(width: 3.5, height: isPlaying ? height : height * 0.86)
      }
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(
      LinearGradient(
        colors: [H3Color.clipAudio.opacity(0.28), Color.black.opacity(0.28)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
      )
    )
  }

  private var playButton: some View {
    Button(action: onTogglePlay) {
      Image(systemName: isPlaying ? "pause.fill" : "play.fill")
        .font(.system(size: 10, weight: .bold))
        .foregroundStyle(.white)
        .offset(x: isPlaying ? 0 : 0.5)
        .frame(width: 24, height: 24)
        .background(isPlaying ? H3Color.accent : Color.black.opacity(0.72))
        .overlay {
          RoundedRectangle(cornerRadius: 6, style: .continuous)
            .stroke(Color.white.opacity(0.18), lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }
    .buttonStyle(.plain)
    .help(isPlaying ? "Pause preview" : "Play preview")
    .accessibilityLabel(isPlaying ? "Pause preview" : "Play preview")
    .accessibilityIdentifier(isPlaying ? "library-pause" : "library-play")
  }

  private var addButton: some View {
    Button {
      model.insertLibraryAsset(asset.id)
    } label: {
      Image(systemName: "plus")
        .font(.system(size: 11, weight: .bold))
        .foregroundStyle(.white)
        .frame(width: 22, height: 22)
        .background(H3Color.accent)
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .shadow(color: H3Color.accent.opacity(0.45), radius: 4, y: 2)
    }
    .buttonStyle(.plain)
    .help("Add to timeline")
    .accessibilityIdentifier("library-insert-\(asset.id.rawValue.uuidString)")
  }

  private func scrub(_ fraction: CGFloat) {
    model.selectedLibraryAssetID = asset.id
    let time = TimeInterval(min(max(fraction, 0), 1)) * max(asset.duration, 0.001)
    currentTime = time
    seekTime = time
    seekNonce += 1
    onEnsurePlaying()
  }

  private static let waveformBars: [CGFloat] = [
    14, 24, 18, 32, 20, 28, 16, 34, 22, 30, 19, 26,
  ]

  static func timecode(_ duration: TimeInterval) -> String {
    let total = max(0, Int(duration.rounded()))
    return String(format: "%d:%02d", total / 60, total % 60)
  }
}

struct LibraryAudioRow: View {
  @Bindable var model: AppModel
  var asset: AssetReference
  var isPlaying: Bool
  var onTogglePlay: () -> Void
  var onEnsurePlaying: () -> Void

  @State private var player = AVPlayer()
  @State private var currentTime: TimeInterval = 0
  @State private var timeObserver: Any?
  @State private var endObserver: NSObjectProtocol?

  private var isSelected: Bool {
    model.selectedLibraryAssetID == asset.id
  }

  private var progress: CGFloat {
    let duration = max(asset.duration, 0.001)
    return min(max(currentTime / duration, 0), 1)
  }

  var body: some View {
    HStack(spacing: 10) {
      playButton
      VStack(alignment: .leading, spacing: 6) {
        HStack(spacing: 8) {
          Text(asset.displayName)
            .font(.system(size: 12, weight: .semibold))
            .lineLimit(1)
          Spacer(minLength: 0)
          Text(LibraryAssetCard.timecode(isPlaying || currentTime > 0.05 ? currentTime : asset.duration))
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .foregroundStyle(H3Color.textSecondary)
        }
        AudioPreviewTimeline(
          progress: progress,
          bars: Self.bars,
          onScrub: scrub
        )
      }
      addButton
    }
    .padding(10)
    .background(H3Color.chrome)
    .overlay {
      RoundedRectangle(cornerRadius: 8, style: .continuous)
        .stroke(isSelected ? H3Color.accent : H3Color.line, lineWidth: isSelected ? 1.5 : 1)
    }
    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    .contentShape(Rectangle())
    .onTapGesture {
      model.selectedLibraryAssetID = asset.id
    }
    .draggable(asset.id.rawValue.uuidString)
    .contextMenu {
      Button("Add to timeline") { model.insertLibraryAsset(asset.id) }
      Button("Download") { AssetFileExporter.presentDownload(of: asset) }
      Button("Rename…") { model.openRail(.adjust) }
      if model.project.usageCount(of: asset.id) == 0 {
        Button("Delete", role: .destructive) {
          model.removeLibraryAsset(asset.id)
        }
      } else {
        Button("Delete and remove clips", role: .destructive) {
          model.removeLibraryAsset(asset.id, removingClips: true)
        }
      }
    }
    .onAppear {
      if isPlaying { attach(); player.play() }
    }
    .onDisappear(perform: detach)
    .onChange(of: isPlaying) { _, playing in
      if playing {
        attach()
        player.play()
      } else {
        player.pause()
      }
    }
    .onChange(of: asset.id) { _, _ in
      detach()
      currentTime = 0
      if isPlaying {
        attach()
        player.play()
      }
    }
    .accessibilityIdentifier("asset-card")
  }

  private var playButton: some View {
    Button(action: onTogglePlay) {
      Image(systemName: isPlaying ? "pause.fill" : "play.fill")
        .font(.system(size: 13, weight: .bold))
        .foregroundStyle(.white)
        .offset(x: isPlaying ? 0 : 0.5)
        .frame(width: 32, height: 32)
        .background(isPlaying ? H3Color.accent : Color.white.opacity(0.14))
        .clipShape(Circle())
    }
    .buttonStyle(.plain)
    .help(isPlaying ? "Pause preview" : "Play preview")
    .accessibilityLabel(isPlaying ? "Pause preview" : "Play preview")
    .accessibilityIdentifier(isPlaying ? "library-pause" : "library-play")
  }

  private var addButton: some View {
    Button {
      model.insertLibraryAsset(asset.id)
    } label: {
      Image(systemName: "plus")
        .font(.system(size: 11, weight: .bold))
        .foregroundStyle(.white)
        .frame(width: 22, height: 22)
        .background(H3Color.accent)
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }
    .buttonStyle(.plain)
    .help("Add to timeline")
    .accessibilityIdentifier("library-insert-\(asset.id.rawValue.uuidString)")
  }

  private func scrub(_ fraction: CGFloat) {
    model.selectedLibraryAssetID = asset.id
    let duration = max(asset.duration, 0.001)
    let time = TimeInterval(min(max(fraction, 0), 1)) * duration
    currentTime = time
    onEnsurePlaying()
    attach()
    player.seek(
      to: CMTime(seconds: time, preferredTimescale: 600),
      toleranceBefore: .zero,
      toleranceAfter: .zero
    )
    player.play()
  }

  private func attach() {
    if player.currentItem == nil {
      player.replaceCurrentItem(with: AVPlayerItem(url: asset.url))
    }
    player.isMuted = false
    if timeObserver == nil {
      timeObserver = player.addPeriodicTimeObserver(
        forInterval: CMTime(seconds: 0.05, preferredTimescale: 600),
        queue: .main
      ) { time in
        MainActor.assumeIsolated {
          currentTime = time.seconds.isFinite ? time.seconds : 0
        }
      }
    }
    if endObserver == nil {
      endObserver = NotificationCenter.default.addObserver(
        forName: .AVPlayerItemDidPlayToEndTime,
        object: player.currentItem,
        queue: .main
      ) { _ in
        MainActor.assumeIsolated {
          player.seek(to: .zero)
          player.play()
          currentTime = 0
        }
      }
    }
  }

  private func detach() {
    player.pause()
    if let timeObserver {
      player.removeTimeObserver(timeObserver)
      self.timeObserver = nil
    }
    if let endObserver {
      NotificationCenter.default.removeObserver(endObserver)
      self.endObserver = nil
    }
    player.replaceCurrentItem(with: nil)
  }

  private static let bars: [CGFloat] = [
    8, 14, 10, 20, 12, 18, 9, 22, 11, 16, 10, 24, 13, 19, 8, 21,
    12, 17, 9, 23, 14, 18, 10, 16, 11, 20, 9, 15, 12, 22, 10, 18,
  ]
}

struct AudioPreviewTimeline: View {
  var progress: CGFloat
  var bars: [CGFloat]
  var onScrub: (CGFloat) -> Void

  var body: some View {
    GeometryReader { proxy in
      let count = max(20, Int(proxy.size.width / 4.5))
      let playhead = min(max(progress, 0), 1) * proxy.size.width
      ZStack(alignment: .leading) {
        HStack(alignment: .center, spacing: 2) {
          ForEach(0..<count, id: \.self) { index in
            let x = (CGFloat(index) + 0.5) * (proxy.size.width / CGFloat(count))
            Capsule()
              .fill(H3Color.accent.opacity(x <= playhead ? 0.95 : 0.32))
              .frame(width: 2.5, height: bars[index % bars.count])
          }
        }
        .frame(width: proxy.size.width, height: proxy.size.height, alignment: .leading)
        Rectangle()
          .fill(Color.white.opacity(0.92))
          .frame(width: 2, height: proxy.size.height)
          .offset(x: max(0, playhead - 1))
      }
      .contentShape(Rectangle())
      .gesture(
        DragGesture(minimumDistance: 0)
          .onChanged { value in
            let width = max(proxy.size.width, 1)
            onScrub(value.location.x / width)
          }
      )
      .accessibilityIdentifier("audio-preview-timeline")
      .accessibilityLabel("Audio preview timeline")
      .accessibilityValue("\(Int((progress * 100).rounded())) percent")
    }
    .frame(height: 28)
  }
}

struct LibraryProgressStrip: View {
  var progress: CGFloat
  var onScrub: (CGFloat) -> Void

  var body: some View {
    GeometryReader { proxy in
      let width = max(proxy.size.width * min(max(progress, 0), 1), 0)
      ZStack(alignment: .bottomLeading) {
        Color.clear
        Capsule()
          .fill(Color.white.opacity(0.22))
          .frame(height: 3)
        Capsule()
          .fill(H3Color.accent)
          .frame(width: max(width, progress > 0 ? 3 : 0), height: 3)
      }
      .padding(.horizontal, 5)
      .padding(.bottom, 3)
      .contentShape(Rectangle())
      .gesture(
        DragGesture(minimumDistance: 0)
          .onChanged { value in
            let track = max(proxy.size.width - 10, 1)
            onScrub((value.location.x - 5) / track)
          }
      )
      .accessibilityIdentifier("video-preview-progress")
      .accessibilityLabel("Video preview progress")
      .accessibilityValue("\(Int((progress * 100).rounded())) percent")
    }
    .frame(height: 10)
  }
}

struct AspectRatioGlyph: View {
  var ratio: CGFloat

  var body: some View {
    let capped = min(max(ratio, 0.42), 2.4)
    let long: CGFloat = 13
    let short = long / max(capped, 1 / capped)
    let width = capped >= 1 ? long : short
    let height = capped >= 1 ? short : long
    RoundedRectangle(cornerRadius: 2, style: .continuous)
      .stroke(H3Color.textSecondary, lineWidth: 1.15)
      .frame(width: width, height: height)
      .accessibilityLabel("Aspect \(Self.label(for: capped))")
  }

  private static func label(for ratio: CGFloat) -> String {
    if abs(ratio - 1) < 0.08 { return "1:1" }
    if ratio > 1 { return String(format: "%.2f:1", ratio) }
    return String(format: "1:%.2f", 1 / ratio)
  }
}
