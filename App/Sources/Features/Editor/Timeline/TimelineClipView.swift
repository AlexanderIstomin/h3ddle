import AppKit
import AVFoundation
import H3ddleCore
import H3ddleDesignSystem
import H3ddleMedia
import ImageIO
import SwiftUI

struct TimelineClipView: View {
  var title: String
  var kind: MediaKind
  var mediaURL: URL? = nil
  var sourceOffset: TimeInterval = 0
  var duration: TimeInterval
  var isEnabled: Bool
  var isTrackMuted: Bool = false
  var isSelected: Bool
  var metrics: TimelineMetrics
  var height: CGFloat
  var accentOverride: Color? = nil
  var showsFilmstrip: Bool = true
  var showsTrimHandles: Bool = false
  var onTrimChanged: ((TimelineTrimEdge, CGFloat) -> Void)?
  var onTrimEnded: (() -> Void)?
  var onMoved: ((CGFloat) -> Void)?
  var onMoveEnded: (() -> Void)?

  private var clipWidth: CGFloat {
    max(8, metrics.x(for: duration) - 2)
  }

  var body: some View {
    ZStack(alignment: .bottomLeading) {
      clipBody
      if showsTrimHandles {
        trimHandle(.leading)
        trimHandle(.trailing)
      }
    }
    .frame(width: clipWidth, height: height)
    .zIndex(isSelected ? 4 : 0)
  }

  private var clipBody: some View {
    ZStack(alignment: .bottomLeading) {
      clipFill
      decoration
      Text(title)
        .font(.system(size: 9, weight: .semibold, design: .monospaced))
        .foregroundStyle(Color.white)
        .lineLimit(1)
        .shadow(color: .black.opacity(0.85), radius: 2, y: 1)
        .padding(.horizontal, 8)
        .padding(.bottom, 4)
    }
    .frame(width: clipWidth, height: height)
    .contentShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
    .gesture(moveDrag)
    .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
    .overlay {
      RoundedRectangle(cornerRadius: 4, style: .continuous)
        .stroke(isSelected ? H3Color.accent : Color.white.opacity(0.2), lineWidth: isSelected ? 1.5 : 1)
    }
    .overlay(alignment: .top) {
      Rectangle()
        .fill(isSelected ? H3Color.accent : accent)
        .frame(height: 2)
        .opacity(isEnabled ? 1 : 0)
        .overlay {
          if !isEnabled {
            Rectangle()
              .stroke(style: StrokeStyle(lineWidth: 2, dash: [4, 3]))
              .foregroundStyle(isSelected ? H3Color.accent : accent.opacity(0.7))
          }
        }
    }
    .overlay {
      if !isEnabled {
        TimelineDisabledHatch()
          .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
      } else if isTrackMuted {
        Color.black.opacity(0.36)
          .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
      }
    }
    .shadow(color: isSelected ? H3Color.accent.opacity(0.35) : .black.opacity(0.45), radius: isSelected ? 8 : 4, y: 3)
    .saturation(clipSaturation)
    .brightness(clipBrightness)
    .opacity(clipOpacity)
    .grayscale(isEnabled ? 0 : 0.68)
  }

  private var clipSaturation: Double {
    if !isEnabled { return 0.2 }
    if isTrackMuted { return 0.42 }
    return 1
  }

  private var clipBrightness: Double {
    if !isEnabled { return -0.18 }
    if isTrackMuted { return -0.12 }
    return 0
  }

  private var clipOpacity: Double {
    if !isEnabled { return 0.68 }
    if isTrackMuted { return 0.78 }
    return 1
  }

  private func trimHandle(_ edge: TimelineTrimEdge) -> some View {
    TimelineTrimHandle()
      .offset(x: edge == .leading ? -10 : clipWidth - 2)
      .highPriorityGesture(
        DragGesture(minimumDistance: 1, coordinateSpace: .global)
          .onChanged { value in
            onTrimChanged?(edge, value.translation.width)
          }
          .onEnded { _ in
            onTrimEnded?()
          }
      )
      .onHover { hovering in
        if hovering {
          NSCursor.resizeLeftRight.push()
        } else {
          NSCursor.pop()
        }
      }
      .accessibilityLabel(edge == .leading ? "Trim start" : "Trim end")
  }

  @ViewBuilder
  private var clipFill: some View {
    if !showsFilmstrip {
      LinearGradient(
        colors: [
          Color(red: 72 / 255, green: 54 / 255, blue: 32 / 255),
          Color(red: 48 / 255, green: 36 / 255, blue: 22 / 255),
        ],
        startPoint: .top,
        endPoint: .bottom
      )
    } else {
      switch kind {
      case .audio:
        LinearGradient(
          colors: [
            Color(red: 44 / 255, green: 58 / 255, blue: 54 / 255),
            Color(red: 31 / 255, green: 41 / 255, blue: 37 / 255),
          ],
          startPoint: .top,
          endPoint: .bottom
        )
      case .video, .image:
        LinearGradient(
          colors: [
            Color(red: 51 / 255, green: 61 / 255, blue: 73 / 255),
            Color(red: 34 / 255, green: 42 / 255, blue: 51 / 255),
          ],
          startPoint: .top,
          endPoint: .bottom
        )
      }
    }
  }

  @ViewBuilder
  private var decoration: some View {
    if !showsFilmstrip {
      EmptyView()
    } else {
    switch kind {
    case .video, .image:
      visualFilmstrip
    case .audio:
      waveform
    }
    }
  }

  private var visualFilmstrip: some View {
    ZStack {
      if let mediaURL {
        TimelineFilmstripView(
          request: TimelineFilmstripRequest(
            url: mediaURL,
            kind: kind,
            sourceOffset: sourceOffset,
            duration: duration
          )
        )
      }
      LinearGradient(
        colors: [.clear, .black.opacity(0.5)],
        startPoint: .center,
        endPoint: .bottom
      )
      filmSprockets.opacity(0.62)
    }
    .allowsHitTesting(false)
  }

  private var filmSprockets: some View {
    ZStack {
      Rectangle()
        .fill(
          .linearGradient(
            colors: [Color.white.opacity(0.1), .clear],
            startPoint: .leading,
            endPoint: .trailing
          )
        )
        .mask(
          Rectangle()
            .stroke(style: StrokeStyle(lineWidth: 1, dash: [1, 17]))
        )
      VStack {
        sprocketRow
        Spacer()
        sprocketRow
      }
    }
  }

  private var sprocketRow: some View {
    Rectangle()
      .fill(Color.black.opacity(0.5))
      .frame(height: 4)
      .mask(
        Rectangle()
          .stroke(style: StrokeStyle(lineWidth: 4, dash: [3, 6]))
      )
  }

  private var waveform: some View {
    Canvas { context, size in
      let barWidth: CGFloat = 2
      let count = max(2, Int(size.width / 4))
      let step = (size.width - barWidth) / CGFloat(count - 1)
      var seed = title.unicodeScalars.reduce(into: UInt64(2_166_136_261)) { partial, scalar in
        partial = partial &* 16_777_619 &+ UInt64(scalar.value)
      }
      for index in 0..<count {
        seed = seed &* 1_664_525 &+ 1_013_904_223
        let value = CGFloat(Double((seed >> 16) & 255) / 255)
        let barHeight = max(4, 28 * (0.18 + value * 0.82))
        let rect = CGRect(
          x: CGFloat(index) * step,
          y: (size.height - barHeight) / 2,
          width: barWidth,
          height: barHeight
        )
        context.fill(
          Path(roundedRect: rect, cornerRadius: barWidth / 2),
          with: .color(H3Color.clipAudio.opacity(0.85))
        )
      }
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }

  private var accent: Color {
    if let accentOverride { return accentOverride }
    return kind == .audio ? H3Color.clipAudio : H3Color.clipVideo
  }

  private var moveDrag: some Gesture {
    DragGesture(minimumDistance: 5, coordinateSpace: .global)
      .onChanged { value in
        onMoved?(value.translation.width)
      }
      .onEnded { _ in
        onMoveEnded?()
      }
  }
}

private struct TimelineFilmstripRequest: Hashable, Sendable {
  var url: URL
  var kind: MediaKind
  var sourceOffset: TimeInterval
  var duration: TimeInterval

  var cacheKey: String {
    if kind == .image {
      return "image|\(url.standardizedFileURL.path)"
    }
    return "video|\(url.standardizedFileURL.path)|\(sourceOffset.bitPattern)|\(duration.bitPattern)"
  }
}

private final class TimelineFilmstripFrames: @unchecked Sendable {
  let images: [CGImage]

  init(_ images: [CGImage]) {
    self.images = images
  }

  var cacheCost: Int {
    images.reduce(0) { cost, image in
      cost + image.bytesPerRow * image.height
    }
  }
}

private actor TimelineFilmstripStore {
  static let shared = TimelineFilmstripStore()

  private let cache = NSCache<NSString, TimelineFilmstripFrames>()
  private var inFlight: [String: Task<TimelineFilmstripFrames?, Never>] = [:]

  private init() {
    cache.countLimit = 64
    cache.totalCostLimit = 128 * 1_024 * 1_024
  }

  func frames(for request: TimelineFilmstripRequest) async -> [CGImage] {
    let key = request.cacheKey
    if let cached = cache.object(forKey: key as NSString) {
      return cached.images
    }
    let task: Task<TimelineFilmstripFrames?, Never>
    if let existing = inFlight[key] {
      task = existing
    } else {
      let created = Task.detached(priority: .utility) {
        await Self.load(request)
      }
      inFlight[key] = created
      task = created
    }
    let loaded = await task.value
    inFlight[key] = nil
    if let loaded {
      cache.setObject(loaded, forKey: key as NSString, cost: loaded.cacheCost)
      return loaded.images
    }
    return []
  }

  private nonisolated static func load(
    _ request: TimelineFilmstripRequest
  ) async -> TimelineFilmstripFrames? {
    guard FileManager.default.fileExists(atPath: request.url.path) else { return nil }
    switch request.kind {
    case .image:
      return loadImage(request.url).map { TimelineFilmstripFrames([$0]) }
    case .video:
      return await loadVideo(request)
    case .audio:
      return nil
    }
  }

  private nonisolated static func loadImage(_ url: URL) -> CGImage? {
    autoreleasepool {
      let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
      guard let source = CGImageSourceCreateWithURL(url as CFURL, sourceOptions) else {
        return nil
      }
      let thumbnailOptions: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: 240,
        kCGImageSourceShouldCacheImmediately: true,
      ]
      return CGImageSourceCreateThumbnailAtIndex(source, 0, thumbnailOptions as CFDictionary)
    }
  }

  private nonisolated static func loadVideo(
    _ request: TimelineFilmstripRequest
  ) async -> TimelineFilmstripFrames? {
    let generator = AVAssetImageGenerator(asset: AVURLAsset(url: request.url))
    generator.appliesPreferredTrackTransform = true
    generator.maximumSize = CGSize(width: 240, height: 144)
    let tolerance = CMTime(seconds: 0.2, preferredTimescale: 600)
    generator.requestedTimeToleranceBefore = tolerance
    generator.requestedTimeToleranceAfter = tolerance
    var images: [CGImage] = []
    for seconds in TimelineFilmstripSampling.times(
      sourceOffset: request.sourceOffset,
      duration: request.duration
    ) {
      let time = CMTime(seconds: seconds, preferredTimescale: 600)
      if let result = try? await generator.image(at: time) {
        images.append(result.image)
      }
    }
    return images.isEmpty ? nil : TimelineFilmstripFrames(images)
  }
}

private struct TimelineFilmstripView: View {
  var request: TimelineFilmstripRequest
  @State private var frames: [CGImage] = []

  var body: some View {
    GeometryReader { proxy in
      if !frames.isEmpty {
        let availableCount = request.kind == .image
          ? TimelineFilmstripSampling.maximumFrameCount
          : frames.count
        let cellCount = min(
          availableCount,
          max(1, Int(ceil(proxy.size.width / 56)))
        )
        let spacing = CGFloat(1)
        let cellWidth = max(
          1,
          (proxy.size.width - spacing * CGFloat(cellCount - 1)) / CGFloat(cellCount)
        )
        HStack(spacing: spacing) {
          ForEach(0..<cellCount, id: \.self) { index in
            Image(decorative: frame(at: index, displayedCount: cellCount), scale: 1)
              .resizable()
              .scaledToFill()
              .frame(width: cellWidth, height: proxy.size.height)
              .clipped()
          }
        }
      }
    }
    .task(id: request) {
      frames = []
      do {
        try await Task.sleep(for: .milliseconds(120))
      } catch {
        return
      }
      let loaded = await TimelineFilmstripStore.shared.frames(for: request)
      guard !Task.isCancelled else { return }
      frames = loaded
    }
  }

  private func frame(at index: Int, displayedCount: Int) -> CGImage {
    if request.kind == .image || frames.count == 1 {
      return frames[0]
    }
    guard displayedCount > 1 else { return frames[frames.count / 2] }
    let position = Double(index) / Double(displayedCount - 1)
    let sourceIndex = Int((position * Double(frames.count - 1)).rounded())
    return frames[sourceIndex]
  }
}

private struct TimelineDisabledHatch: View {
  var body: some View {
    Canvas { context, size in
      let spacing: CGFloat = 8
      var x: CGFloat = -size.height
      while x < size.width + size.height {
        var path = Path()
        path.move(to: CGPoint(x: x, y: size.height))
        path.addLine(to: CGPoint(x: x + size.height, y: 0))
        context.stroke(path, with: .color(.black.opacity(0.22)), lineWidth: 1)
        x += spacing
      }
    }
    .background(Color.black.opacity(0.46))
    .allowsHitTesting(false)
  }
}

private struct TimelineTrimHandle: View {
  var body: some View {
    ZStack {
      Color.clear
        .frame(width: 12)
      Capsule()
        .fill(H3Color.accent.opacity(0.92))
        .frame(width: 2, height: 18)
        .shadow(color: .black.opacity(0.45), radius: 1, y: 1)
    }
    .frame(width: 12)
    .frame(maxHeight: .infinity)
    .padding(.vertical, -4)
    .contentShape(Rectangle())
  }
}
