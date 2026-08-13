import AVFoundation
import H3ddleDesignSystem
import SwiftUI

struct AudioPreviewPlayer: View {
  let url: URL
  let duration: TimeInterval

  @State private var player = AVPlayer()
  @State private var isPlaying = false
  @State private var currentTime: TimeInterval = 0
  @State private var observer: Any?
  @State private var endObserver: NSObjectProtocol?

  var body: some View {
    VStack(spacing: 16) {
      Image(systemName: "waveform")
        .font(.system(size: 28, weight: .medium))
        .foregroundStyle(H3Color.clipAudio)
      Text(url.lastPathComponent)
        .font(.system(size: 12, weight: .medium))
        .lineLimit(1)
        .foregroundStyle(H3Color.textSecondary)

      HStack(spacing: 12) {
        Button {
          toggle()
        } label: {
          Image(systemName: isPlaying ? "pause.fill" : "play.fill")
            .font(.system(size: 16, weight: .semibold))
            .foregroundStyle(Color.white)
            .frame(width: 40, height: 40)
            .background(H3Color.accent)
            .clipShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(!fileExists)
        .accessibilityIdentifier("audio-preview-play")

        VStack(alignment: .leading, spacing: 4) {
          ProgressView(value: progress)
            .tint(H3Color.accent)
          HStack {
            Text(Self.format(currentTime))
            Spacer()
            Text(Self.format(duration))
          }
          .font(.system(size: 10, weight: .medium, design: .monospaced))
          .foregroundStyle(H3Color.textSecondary)
        }
      }

      if !fileExists {
        Text("Audio file is missing or unreadable.")
          .font(.system(size: 11, weight: .medium))
          .foregroundStyle(H3Color.danger)
      }
    }
    .padding(20)
    .onAppear(perform: attach)
    .onDisappear(perform: detach)
    .onChange(of: url) { _, _ in
      detach()
      attach()
    }
  }

  private var fileExists: Bool {
    FileManager.default.fileExists(atPath: url.path)
  }

  private var progress: Double {
    guard duration > 0 else { return 0 }
    return min(max(currentTime / duration, 0), 1)
  }

  private func toggle() {
    if isPlaying {
      player.pause()
      isPlaying = false
    } else {
      if currentTime >= duration - 0.05 {
        player.seek(to: .zero)
        currentTime = 0
      }
      player.play()
      isPlaying = true
    }
  }

  private func attach() {
    guard fileExists else { return }
    player.replaceCurrentItem(with: AVPlayerItem(url: url))
    // Both callbacks are delivered on the main queue; assumeIsolated makes
    // that visible to strict-concurrency compilers.
    observer = player.addPeriodicTimeObserver(
      forInterval: CMTime(seconds: 0.1, preferredTimescale: 600),
      queue: .main
    ) { time in
      MainActor.assumeIsolated {
        currentTime = time.seconds.isFinite ? time.seconds : 0
      }
    }
    endObserver = NotificationCenter.default.addObserver(
      forName: .AVPlayerItemDidPlayToEndTime,
      object: player.currentItem,
      queue: .main
    ) { _ in
      MainActor.assumeIsolated {
        isPlaying = false
        currentTime = duration
      }
    }
  }

  private func detach() {
    player.pause()
    isPlaying = false
    if let observer {
      player.removeTimeObserver(observer)
      self.observer = nil
    }
    if let endObserver {
      NotificationCenter.default.removeObserver(endObserver)
      self.endObserver = nil
    }
    player.replaceCurrentItem(with: nil)
  }

  private static func format(_ time: TimeInterval) -> String {
    let total = max(0, Int(time.rounded()))
    return String(format: "%d:%02d", total / 60, total % 60)
  }
}
