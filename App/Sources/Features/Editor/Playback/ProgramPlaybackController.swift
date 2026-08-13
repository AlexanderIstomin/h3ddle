import AVFoundation
import H3ddleCore
import H3ddleMedia
import Observation

@MainActor
@Observable
final class ProgramPlaybackController {
  var clock = ProgramClock()
  let visualPlayer = AVPlayer()
  let audioPlayer = AVPlayer()

  private var tickTask: Task<Void, Never>?
  private var currentVisualURL: URL?
  private var currentAudioURL: URL?
  private var lastVideoLocalTime: TimeInterval = 0

  var isPlaying: Bool { clock.isPlaying }

  func toggle(duration: TimeInterval) {
    if clock.isPlaying {
      pause()
    } else {
      play(duration: duration)
    }
  }

  func play(duration: TimeInterval) {
    guard duration > 0 else { return }
    if clock.currentTime >= duration {
      clock.skipToStart()
    }
    clock.isPlaying = true
    startTicking(duration: duration)
    if visualPlayer.currentItem != nil {
      visualPlayer.play()
    }
    if audioPlayer.currentItem != nil {
      audioPlayer.play()
    }
  }

  func pause() {
    clock.isPlaying = false
    tickTask?.cancel()
    tickTask = nil
    visualPlayer.pause()
    audioPlayer.pause()
  }

  func seek(_ time: TimeInterval, duration: TimeInterval) {
    clock.setTime(time, duration: duration)
    if clock.isPlaying {
      visualPlayer.play()
      audioPlayer.play()
    }
  }

  func step(frames: Int, duration: TimeInterval) {
    pause()
    clock.step(frames: frames, duration: duration)
  }

  func skipToStart() {
    clock.skipToStart()
  }

  func skipToEnd(duration: TimeInterval) {
    clock.skipToEnd(duration: duration)
    if clock.currentTime >= duration {
      pause()
    }
  }

  func sync(
    project: H3ddleProject,
    visualMuted: Bool,
    audioMuted: Bool
  ) {
    let duration = max(project.timeline.visualDuration, project.timeline.audioTrackEnd)
    clock.framesPerSecond = project.settings.framesPerSecond
    clock.setTime(clock.currentTime, duration: duration)
    applyMasterGain(project.settings.masterGain)
    let frame = ProgramPreview.frame(
      at: clock.currentTime,
      project: project,
      visualMuted: visualMuted,
      audioMuted: audioMuted
    )
    applyVisual(frame)
    applyAudio(frame)
  }

  func shutdown() {
    pause()
    visualPlayer.replaceCurrentItem(with: nil)
    audioPlayer.replaceCurrentItem(with: nil)
    currentVisualURL = nil
    currentAudioURL = nil
  }

  private func startTicking(duration: TimeInterval) {
    tickTask?.cancel()
    tickTask = Task { [weak self] in
      while !Task.isCancelled {
        do {
          try await Task.sleep(for: .milliseconds(33))
        } catch {
          return
        }
        guard let self, self.clock.isPlaying else { return }
        if !self.clock.advance(by: 1.0 / 30.0, duration: duration) {
          self.pause()
          return
        }
      }
    }
  }

  private func applyMasterGain(_ gain: Double) {
    let volume = Float(min(max(gain, 0), 1))
    visualPlayer.volume = volume
    audioPlayer.volume = volume
  }

  private func applyVisual(_ frame: ProgramPreviewFrame) {
    switch frame.visual {
    case .video(let asset, let localTime, let includesNativeAudio):
      visualPlayer.isMuted = !includesNativeAudio
      let replaced = replaceVisualItemIfNeeded(url: asset.url)
      let loopedBack = localTime + 0.2 < lastVideoLocalTime
      lastVideoLocalTime = localTime
      if replaced || !clock.isPlaying || loopedBack {
        seek(player: visualPlayer, to: localTime)
      }
      if clock.isPlaying, visualPlayer.rate == 0 {
        visualPlayer.play()
      }
    case .image, .empty:
      lastVideoLocalTime = 0
      visualPlayer.pause()
      if currentVisualURL != nil {
        currentVisualURL = nil
        visualPlayer.replaceCurrentItem(with: nil)
      }
    }
  }

  private func applyAudio(_ frame: ProgramPreviewFrame) {
    guard let presentation = frame.audio.first,
      FileManager.default.fileExists(atPath: presentation.asset.url.path)
    else {
      audioPlayer.pause()
      if currentAudioURL != nil {
        currentAudioURL = nil
        audioPlayer.replaceCurrentItem(with: nil)
      }
      return
    }

    let replaced = currentAudioURL != presentation.asset.url
    if replaced {
      currentAudioURL = presentation.asset.url
      audioPlayer.replaceCurrentItem(with: AVPlayerItem(url: presentation.asset.url))
    }
    if replaced || !clock.isPlaying {
      let shouldResume = clock.isPlaying
      seek(player: audioPlayer, to: presentation.localTime, resumeIfPlaying: shouldResume)
      return
    }
    if clock.isPlaying, audioPlayer.rate == 0 {
      audioPlayer.play()
    }
  }

  private func replaceVisualItemIfNeeded(url: URL) -> Bool {
    guard currentVisualURL != url else { return false }
    currentVisualURL = url
    visualPlayer.replaceCurrentItem(with: AVPlayerItem(url: url))
    return true
  }

  private func seek(
    player: AVPlayer,
    to seconds: TimeInterval,
    resumeIfPlaying: Bool = false
  ) {
    let time = CMTime(seconds: max(0, seconds), preferredTimescale: 600)
    player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] _ in
      guard resumeIfPlaying else { return }
      Task { @MainActor in
        guard let self, self.clock.isPlaying else { return }
        player.play()
      }
    }
  }
}
