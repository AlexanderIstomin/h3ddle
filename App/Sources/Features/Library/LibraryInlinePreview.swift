import AVFoundation
import CoreGraphics
import H3ddleCore
import SwiftUI

/// In-card looping preview. Reports time so the card can draw a progress strip.
struct LibraryLoopingPlayer: NSViewRepresentable {
  var url: URL
  var isMuted: Bool
  var isPlaying: Bool
  var seekNonce: UInt
  var seekTime: TimeInterval
  var onTime: (TimeInterval) -> Void

  func makeNSView(context: Context) -> LibraryPlayerNSView {
    let view = LibraryPlayerNSView()
    context.coordinator.callbacks.onTime = onTime
    context.coordinator.callbacks.isPlaying = isPlaying
    context.coordinator.attach(url: url, muted: isMuted, to: view)
    context.coordinator.applySeek(seekNonce: seekNonce, time: seekTime, on: view)
    if isPlaying { view.player?.play() }
    return view
  }

  func updateNSView(_ view: LibraryPlayerNSView, context: Context) {
    context.coordinator.callbacks.onTime = onTime
    context.coordinator.callbacks.isPlaying = isPlaying
    let current = (view.player?.currentItem?.asset as? AVURLAsset)?.url
    if current?.standardizedFileURL != url.standardizedFileURL {
      context.coordinator.attach(url: url, muted: isMuted, to: view)
    }
    view.player?.isMuted = isMuted
    context.coordinator.applySeek(seekNonce: seekNonce, time: seekTime, on: view)
    if isPlaying {
      view.player?.play()
    } else {
      view.player?.pause()
    }
  }

  static func dismantleNSView(_ view: LibraryPlayerNSView, coordinator: Coordinator) {
    coordinator.detach(view)
  }

  func makeCoordinator() -> Coordinator {
    Coordinator()
  }

  @MainActor
  final class Coordinator {
    let callbacks = Callbacks()
    private var loopObserver: NSObjectProtocol?
    private var timeObserver: Any?
    private var appliedSeekNonce: UInt = 0

    final class Callbacks: @unchecked Sendable {
      var onTime: (TimeInterval) -> Void = { _ in }
      var isPlaying = false
    }

    func attach(url: URL, muted: Bool, to view: LibraryPlayerNSView) {
      detach(view)
      let player = AVPlayer(url: url)
      player.isMuted = muted
      view.player = player
      let callbacks = callbacks
      timeObserver = player.addPeriodicTimeObserver(
        forInterval: CMTime(seconds: 0.05, preferredTimescale: 600),
        queue: .main
      ) { time in
        let seconds = time.seconds.isFinite ? time.seconds : 0
        callbacks.onTime(seconds)
      }
      loopObserver = NotificationCenter.default.addObserver(
        forName: .AVPlayerItemDidPlayToEndTime,
        object: player.currentItem,
        queue: .main
      ) { [weak player] _ in
        guard callbacks.isPlaying else { return }
        player?.seek(to: .zero)
        player?.play()
      }
    }

    func applySeek(seekNonce: UInt, time: TimeInterval, on view: LibraryPlayerNSView) {
      guard seekNonce != 0, seekNonce != appliedSeekNonce else { return }
      appliedSeekNonce = seekNonce
      view.player?.seek(
        to: CMTime(seconds: max(0, time), preferredTimescale: 600),
        toleranceBefore: .zero,
        toleranceAfter: .zero
      )
    }

    func detach(_ view: LibraryPlayerNSView) {
      if let timeObserver {
        view.player?.removeTimeObserver(timeObserver)
        self.timeObserver = nil
      }
      if let loopObserver {
        NotificationCenter.default.removeObserver(loopObserver)
        self.loopObserver = nil
      }
      view.player?.pause()
      view.player = nil
    }
  }
}

final class LibraryPlayerNSView: NSView {
  let playerLayer = AVPlayerLayer()
  var player: AVPlayer? {
    didSet { playerLayer.player = player }
  }

  override init(frame frameRect: NSRect) {
    super.init(frame: frameRect)
    wantsLayer = true
    clipsToBounds = true
    playerLayer.videoGravity = .resizeAspectFill
    layer?.addSublayer(playerLayer)
  }

  required init?(coder: NSCoder) {
    nil
  }

  override func layout() {
    super.layout()
    playerLayer.frame = bounds
  }
}

struct LibraryVideoPoster: View {
  var url: URL
  @State private var image: CGImage?

  var body: some View {
    Group {
      if let image {
        Image(decorative: image, scale: 1)
          .resizable()
          .interpolation(.high)
          .scaledToFill()
      } else {
        Color.clear
      }
    }
    .task(id: url) {
      image = await LibraryPosterCache.shared.poster(for: url)
    }
  }
}

actor LibraryPosterCache {
  static let shared = LibraryPosterCache()

  private var images: [URL: CGImage] = [:]

  func poster(for url: URL) async -> CGImage? {
    if let cached = images[url] { return cached }
    guard let generated = await Self.generate(url) else { return nil }
    images[url] = generated
    return generated
  }

  nonisolated private static func generate(_ url: URL) async -> CGImage? {
    let asset = AVURLAsset(url: url)
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.maximumSize = CGSize(width: 640, height: 640)
    let time = CMTime(seconds: 0.12, preferredTimescale: 600)
    return try? await generator.image(at: time).image
  }
}

enum LibraryMasonry {
  static let columns = 2
  static let gap: CGFloat = 6

  static func pack(
    _ assets: [AssetReference],
    sizes: [AssetID: CGSize]
  ) -> [[AssetReference]] {
    var columns: [[AssetReference]] = Array(repeating: [], count: Self.columns)
    var heights = Array(repeating: CGFloat(0), count: Self.columns)
    for asset in assets {
      let ratio: CGFloat
      if let size = sizes[asset.id], size.width > 1 {
        ratio = size.height / size.width
      } else if asset.kind == .audio {
        ratio = 0.72
      } else {
        ratio = 10 / 16
      }
      let estimated = min(max(ratio, 0.45), 2.2) + 0.22
      var target = 0
      for index in 1..<Self.columns where heights[index] + 0.000_1 < heights[target] {
        target = index
      }
      columns[target].append(asset)
      heights[target] += estimated
    }
    return columns
  }
}
