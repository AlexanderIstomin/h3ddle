import Foundation
import H3ddleCore

public enum ProgramExportPreset: String, CaseIterable, Sendable, Identifiable {
  case recommended
  case high
  case smaller
  case master
  case custom

  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .recommended: "Recommended"
    case .high: "High quality"
    case .smaller: "Smaller file"
    case .master: "Master"
    case .custom: "Custom"
    }
  }

  public var subtitle: String {
    switch self {
    case .recommended: "balanced"
    case .high: "crisp"
    case .smaller: "compact"
    case .master: "ProRes"
    case .custom: "tune all"
    }
  }

  public var symbolName: String {
    switch self {
    case .recommended: "sparkles"
    case .high: "rosette"
    case .smaller: "leaf"
    case .master: "crown"
    case .custom: "slider.horizontal.3"
    }
  }
}

public enum ProgramExportFormat: String, CaseIterable, Sendable, Identifiable {
  case h264
  case h265
  case proRes

  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .h264: "H.264"
    case .h265: "H.265"
    case .proRes: "ProRes"
    }
  }

  public var fileExtension: String {
    switch self {
    case .h264, .h265: "mp4"
    case .proRes: "mov"
    }
  }

  public var containerLabel: String {
    fileExtension.uppercased()
  }
}

public enum ProgramExportProfile: String, CaseIterable, Sendable, Identifiable {
  case auto
  case baseline
  case main
  case high

  public var id: String { rawValue }

  public var label: String { rawValue }
}

public enum ProgramAudioQuality: Int, CaseIterable, Sendable, Identifiable {
  case low = 96
  case medium = 128
  case high = 192
  case max = 320

  public var id: Int { rawValue }

  public var label: String {
    switch self {
    case .low: "Low"
    case .medium: "Medium"
    case .high: "High"
    case .max: "Max"
    }
  }

  public var kilobitsPerSecond: Int { rawValue }
}

public enum ProgramExportRangeMode: String, Sendable {
  case full
  case custom
}

public struct ProgramExportRange: Hashable, Sendable {
  public var mode: ProgramExportRangeMode
  public var inSec: TimeInterval
  public var outSec: TimeInterval

  public init(
    mode: ProgramExportRangeMode = .full,
    inSec: TimeInterval = 0,
    outSec: TimeInterval = 0
  ) {
    self.mode = mode
    self.inSec = max(0, inSec)
    self.outSec = max(self.inSec, outSec)
  }

  public func resolved(
    in duration: TimeInterval,
    framesPerSecond: Double = 24
  ) -> (inSec: TimeInterval, outSec: TimeInterval) {
    let duration = max(0, duration)
    if mode == .full || duration == 0 {
      return (0, duration)
    }
    return ProgramExportSettings.clampRange(
      inSec: inSec,
      outSec: outSec,
      duration: duration,
      framesPerSecond: framesPerSecond
    )
  }

  public func duration(
    in programDuration: TimeInterval,
    framesPerSecond: Double = 24
  ) -> TimeInterval {
    let span = resolved(in: programDuration, framesPerSecond: framesPerSecond)
    return max(0, span.outSec - span.inSec)
  }
}

public struct ProgramExportSettings: Hashable, Sendable {
  public static let minimumVideoBitrateKbps = 200.0
  public static let maximumVideoBitrateKbps = 60_000.0
  public static let videoBitrateStepKbps = 50.0
  public static let loudnessTargetLUFS = -14.0

  public var preset: ProgramExportPreset
  public var format: ProgramExportFormat
  public var resolution: ProjectResolution
  public var framesPerSecond: Double
  public var videoBitrateKbps: Double
  public var profile: ProgramExportProfile
  public var audioBitrateKbps: Int
  public var normalizeLoudness: Bool
  public var range: ProgramExportRange
  public var usesHardwareAcceleration: Bool
  /// When false, A1 and native clip soundtracks are omitted from the file.
  public var includeAudioLane: Bool

  public init(
    preset: ProgramExportPreset = .recommended,
    format: ProgramExportFormat = .h264,
    resolution: ProjectResolution = .fullHD,
    framesPerSecond: Double = 24,
    videoBitrateKbps: Double = 8_000,
    profile: ProgramExportProfile = .auto,
    audioBitrateKbps: Int = 192,
    normalizeLoudness: Bool = false,
    range: ProgramExportRange = ProgramExportRange(),
    usesHardwareAcceleration: Bool = true,
    includeAudioLane: Bool = true
  ) {
    self.preset = preset
    self.format = format
    self.resolution = resolution
    self.framesPerSecond = ProgramExportSettings.nearestFrameRate(framesPerSecond)
    self.videoBitrateKbps = ProgramExportSettings.clampBitrate(videoBitrateKbps)
    self.profile = profile
    self.audioBitrateKbps = ProgramAudioQuality.nearest(to: audioBitrateKbps).kilobitsPerSecond
    self.normalizeLoudness = normalizeLoudness
    self.range = range
    self.usesHardwareAcceleration = usesHardwareAcceleration
    self.includeAudioLane = includeAudioLane
  }

  public static func makeDefault(project: H3ddleProject) -> ProgramExportSettings {
    let settings = project.settings
    let defaults = PlatformExportDefaults.defaults(for: settings.platform)
    return ProgramExportSettings(
      preset: .recommended,
      format: .h264,
      resolution: settings.resolution ?? .fullHD,
      framesPerSecond: nearestFrameRate(settings.framesPerSecond),
      videoBitrateKbps: defaults.videoBitrateKbps,
      profile: .auto,
      audioBitrateKbps: defaults.audioBitrateKbps,
      normalizeLoudness: defaults.normalizeLoudness,
      range: ProgramExportRange(
        mode: .full,
        inSec: 0,
        outSec: project.timeline.visualDuration
      ),
      usesHardwareAcceleration: true
    )
  }

  public var audioQuality: ProgramAudioQuality {
    ProgramAudioQuality.nearest(to: audioBitrateKbps)
  }

  public var isCustom: Bool { preset == .custom }

  public mutating func apply(preset newPreset: ProgramExportPreset, seed: ProgramExportSettings) {
    self = ProgramExportSettings.applying(newPreset, to: self, seed: seed)
  }

  public mutating func updateCustom(_ body: (inout ProgramExportSettings) -> Void) {
    body(&self)
    videoBitrateKbps = ProgramExportSettings.clampBitrate(videoBitrateKbps)
    framesPerSecond = ProgramExportSettings.nearestFrameRate(framesPerSecond)
    audioBitrateKbps = ProgramAudioQuality.nearest(to: audioBitrateKbps).kilobitsPerSecond
    preset = .custom
  }

  public mutating func setAdditiveNormalize(_ on: Bool) {
    normalizeLoudness = on
  }

  public mutating func setAdditiveHardwareAcceleration(_ on: Bool) {
    usesHardwareAcceleration = on
  }

  public func outputPixelSize(project: ProjectSettings) -> (width: Int, height: Int) {
    var canvas = project
    canvas.apply(resolution: resolution)
    return (
      ProgramExportSettings.evenDimension(canvas.width),
      ProgramExportSettings.evenDimension(canvas.height)
    )
  }

  public func rangeDuration(programDuration: TimeInterval) -> TimeInterval {
    range.duration(in: programDuration, framesPerSecond: framesPerSecond)
  }

  public func suggestedFileName(projectName: String) -> String {
    let trimmed = projectName.trimmingCharacters(in: .whitespacesAndNewlines)
    let base = (trimmed.isEmpty ? "H3ddle" : trimmed)
      .replacingOccurrences(of: "/", with: "-")
      .replacingOccurrences(of: ":", with: "-")
    return "\(base)_export.\(format.fileExtension)"
  }

  public func estimatedSizeMegabytes(programDuration: TimeInterval, project: ProjectSettings)
    -> Double
  {
    let duration = max(0, rangeDuration(programDuration: programDuration))
    let videoRate: Double
    if format == .proRes {
      let size = outputPixelSize(project: project)
      videoRate = Double(size.width * size.height) * framesPerSecond * 0.18 / 1_000
    } else {
      videoRate = videoBitrateKbps
    }
    return (videoRate + Double(audioBitrateKbps)) * duration / 8_000
  }

  public func estimatedRenderSeconds(programDuration: TimeInterval) -> Double {
    let duration = max(0, rangeDuration(programDuration: programDuration))
    var seconds = duration * resolution.renderCost
    if usesHardwareAcceleration { seconds /= 2.6 }
    if format == .h265 { seconds *= 1.5 }
    if format == .proRes { seconds *= 1.2 }
    return max(4, seconds)
  }

  public static func applying(
    _ preset: ProgramExportPreset,
    to current: ProgramExportSettings,
    seed: ProgramExportSettings
  ) -> ProgramExportSettings {
    switch preset {
    case .custom:
      var next = current
      next.preset = .custom
      return next
    case .master:
      var next = seed
      next.preset = .master
      next.format = .proRes
      next.resolution = .ultraHD
      next.profile = .high
      next.audioBitrateKbps = ProgramAudioQuality.max.kilobitsPerSecond
      next.normalizeLoudness = current.normalizeLoudness
      next.usesHardwareAcceleration = current.usesHardwareAcceleration
      next.includeAudioLane = current.includeAudioLane
      next.range = current.range
      return next
    case .recommended, .high, .smaller:
      var next = seed
      next.preset = preset
      next.normalizeLoudness = current.normalizeLoudness
      next.usesHardwareAcceleration = current.usesHardwareAcceleration
      next.includeAudioLane = current.includeAudioLane
      next.range = current.range
      let scale: Double = preset == .high ? 1.5 : preset == .smaller ? 0.6 : 1
      next.videoBitrateKbps = clampBitrate(seed.videoBitrateKbps * scale)
      if preset == .smaller {
        next.resolution = seed.resolution.steppedDown()
      }
      return next
    }
  }

  public static func clampBitrate(_ value: Double) -> Double {
    min(maximumVideoBitrateKbps, max(minimumVideoBitrateKbps, value))
  }

  public static func nearestFrameRate(_ value: Double) -> Double {
    ProjectSettings.frameRates.min { abs($0 - value) < abs($1 - value) } ?? 24
  }

  public static func evenDimension(_ value: Int) -> Int {
    max(2, value - value % 2)
  }

  public static func formatClock(_ seconds: TimeInterval) -> String {
    let total = max(0, Int(seconds.rounded()))
    let minutes = total / 60
    let remainder = total % 60
    return "\(minutes):\(String(format: "%02d", remainder))"
  }

  public static func parseClock(_ text: String) -> TimeInterval? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    let parts = trimmed.split(separator: ":")
    if parts.count == 2, let minutes = Double(parts[0]), let seconds = Double(parts[1]) {
      return minutes * 60 + seconds
    }
    return Double(trimmed)
  }

  public static func clampRange(
    inSec: TimeInterval,
    outSec: TimeInterval,
    duration: TimeInterval,
    framesPerSecond: Double
  ) -> (inSec: TimeInterval, outSec: TimeInterval) {
    let duration = max(0, duration)
    let frame = 1 / max(framesPerSecond, 1)
    let lo = min(max(0, inSec), max(0, duration - frame))
    let hi = min(duration, max(outSec, lo + frame))
    return (lo, hi)
  }
}

public struct PlatformExportDefaults: Equatable, Sendable {
  public var videoBitrateKbps: Double
  public var audioBitrateKbps: Int
  public var normalizeLoudness: Bool

  public static func defaults(for platform: ProjectPlatform) -> PlatformExportDefaults {
    switch platform {
    case .youtube, .youtubeShorts:
      PlatformExportDefaults(
        videoBitrateKbps: 8_000,
        audioBitrateKbps: 320,
        normalizeLoudness: true
      )
    case .tiktok:
      PlatformExportDefaults(
        videoBitrateKbps: 10_000,
        audioBitrateKbps: 192,
        normalizeLoudness: true
      )
    case .instagramReel, .instagramStory, .instagramPost, .instagramFeed:
      PlatformExportDefaults(
        videoBitrateKbps: 5_000,
        audioBitrateKbps: 192,
        normalizeLoudness: true
      )
    case .landscape, .portrait, .square, .custom:
      PlatformExportDefaults(
        videoBitrateKbps: 8_000,
        audioBitrateKbps: 192,
        normalizeLoudness: false
      )
    }
  }
}

extension ProgramAudioQuality {
  static func nearest(to kbps: Int) -> ProgramAudioQuality {
    allCases.min { abs($0.rawValue - kbps) < abs($1.rawValue - kbps) } ?? .high
  }
}

extension ProjectResolution {
  var renderCost: Double {
    switch self {
    case .ultraHD: 2.4
    case .fullHD: 0.92
    case .hd: 0.55
    case .sd: 0.35
    case .low: 0.26
    case .veryLow: 0.2
    case .extreme: 0.12
    }
  }

  func steppedDown() -> ProjectResolution {
    let all = ProjectResolution.allCases
    guard let index = all.firstIndex(of: self) else { return self }
    return all[min(index + 1, all.count - 1)]
  }
}