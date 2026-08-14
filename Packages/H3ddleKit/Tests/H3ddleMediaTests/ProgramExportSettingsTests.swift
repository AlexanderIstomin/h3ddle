import Foundation
import H3ddleCore
import Testing

@testable import H3ddleMedia

@Suite("Program export settings")
struct ProgramExportSettingsTests {
  @Test("YouTube platform seeds bitrate and loudness")
  func youtubeSeed() throws {
    var project = H3ddleProject()
    project.settings.apply(platform: .youtube)
    try project.timeline.appendVisual(videoAsset(duration: 12))
    let settings = ProgramExportSettings.makeDefault(project: project)
    #expect(settings.preset == .recommended)
    #expect(settings.format == .h264)
    #expect(settings.resolution == .fullHD)
    #expect(settings.framesPerSecond == 30)
    #expect(settings.videoBitrateKbps == 8_000)
    #expect(settings.audioBitrateKbps == 320)
    #expect(settings.normalizeLoudness)
    #expect(settings.range.mode == .full)
    #expect(settings.range.outSec == 12)
  }

  @Test("Custom platform leaves loudness off")
  func customSeed() {
    let settings = ProgramExportSettings.makeDefault(project: H3ddleProject())
    #expect(settings.framesPerSecond == 24)
    #expect(settings.audioBitrateKbps == 192)
    #expect(!settings.normalizeLoudness)
  }

  @Test("Named presets scale from the project seed")
  func namedPresets() {
    let seed = ProgramExportSettings.makeDefault(project: H3ddleProject())
    var settings = seed

    settings.apply(preset: .high, seed: seed)
    #expect(settings.preset == .high)
    #expect(settings.resolution == .fullHD)
    #expect(abs(settings.videoBitrateKbps - 12_000) < 0.01)

    settings.apply(preset: .smaller, seed: seed)
    #expect(settings.preset == .smaller)
    #expect(settings.resolution == .hd)
    #expect(abs(settings.videoBitrateKbps - 4_800) < 0.01)

    settings.apply(preset: .recommended, seed: seed)
    #expect(settings.preset == .recommended)
    #expect(settings.resolution == .fullHD)
    #expect(abs(settings.videoBitrateKbps - 8_000) < 0.01)
  }

  @Test("Master switches to ProRes 4K without dropping additive toggles")
  func masterPreset() {
    let seed = ProgramExportSettings.makeDefault(project: H3ddleProject())
    var settings = seed
    settings.setAdditiveNormalize(true)
    settings.setAdditiveHardwareAcceleration(false)
    settings.apply(preset: .master, seed: seed)
    #expect(settings.preset == .master)
    #expect(settings.format == .proRes)
    #expect(settings.resolution == .ultraHD)
    #expect(settings.audioBitrateKbps == 320)
    #expect(settings.normalizeLoudness)
    #expect(!settings.usesHardwareAcceleration)
    #expect(settings.format.fileExtension == "mov")
  }

  @Test("Editing a locked field selects Custom")
  func customFromEdit() {
    var settings = ProgramExportSettings.makeDefault(project: H3ddleProject())
    settings.updateCustom { $0.videoBitrateKbps = 3_000 }
    #expect(settings.preset == .custom)
    #expect(settings.videoBitrateKbps == 3_000)
  }

  @Test("Additive loudness does not select Custom")
  func additiveStaysOnPreset() {
    var settings = ProgramExportSettings.makeDefault(project: H3ddleProject())
    settings.setAdditiveNormalize(true)
    #expect(settings.preset == .recommended)
    #expect(settings.normalizeLoudness)
  }

  @Test("Portrait 1080p export stays 1080 wide")
  func portraitOutputSize() {
    var project = H3ddleProject()
    project.settings.apply(platform: .tiktok)
    var settings = ProgramExportSettings.makeDefault(project: project)
    settings.resolution = .fullHD
    let size = settings.outputPixelSize(project: project.settings)
    #expect(size.width == 1080)
    #expect(size.height == 1920)
  }

  @Test("Odd canvas edges become even")
  func evenDimensions() {
    #expect(ProgramExportSettings.evenDimension(213) == 212)
    #expect(ProgramExportSettings.evenDimension(1) == 2)
  }

  @Test("Size estimate uses both bitrates and the export range")
  func sizeEstimate() {
    let settings = ProgramExportSettings(
      videoBitrateKbps: 8_000,
      audioBitrateKbps: 320,
      range: ProgramExportRange(mode: .custom, inSec: 0, outSec: 10)
    )
    let size = settings.estimatedSizeMegabytes(
      programDuration: 20,
      project: .default
    )
    #expect(abs(size - 10.4) < 0.01)
  }

  @Test("Clock helpers round-trip")
  func clocks() {
    #expect(ProgramExportSettings.formatClock(154) == "2:34")
    #expect(ProgramExportSettings.parseClock("2:34") == 154)
    #expect(ProgramExportSettings.parseClock("90") == 90)
  }

  @Test("Custom range keeps at least one frame")
  func clampRange() {
    let clamped = ProgramExportSettings.clampRange(
      inSec: 9.9,
      outSec: 9.91,
      duration: 10,
      framesPerSecond: 24
    )
    // Subtracting the clamped bounds loses the last bits of precision, so
    // allow a tolerance far smaller than a frame.
    #expect(clamped.outSec - clamped.inSec >= 1 / 24 - 1e-9)
  }

  @Test("Trailing audio past the visual duration is a warning")
  func trailingAudioWarning() throws {
    var project = H3ddleProject()
    let visual = videoAsset(duration: 4)
    let audio = AssetReference(
      kind: .audio,
      displayName: "Score",
      url: URL(fileURLWithPath: "/tmp/score.wav"),
      duration: 8
    )
    project.addAsset(visual)
    project.addAsset(audio)
    try project.timeline.appendVisual(visual)
    try project.timeline.appendAudio(audio)

    let plan = ProgramCompositionPlan(project: project)
    let full = ProgramExportRange(mode: .full, inSec: 0, outSec: 4)
    #expect(plan.requiresTrailingAudioWarning(range: full))
    #expect(abs(plan.trailingAudioPast(range: full) - 4) < 0.001)

    project.timeline.setAudioTrim(
      project.timeline.audioItems[0].id,
      AudioTrim(startTime: 0, duration: 3, sourceOffset: 0)
    )
    let shortPlan = ProgramCompositionPlan(project: project)
    #expect(!shortPlan.requiresTrailingAudioWarning(range: full))

    let cropped = ProgramExportRange(mode: .custom, inSec: 0, outSec: 2)
    #expect(shortPlan.requiresTrailingAudioWarning(range: cropped))
  }

  private func videoAsset(duration: TimeInterval) -> AssetReference {
    AssetReference(
      kind: .video,
      displayName: "Clip",
      url: URL(fileURLWithPath: "/tmp/clip.mp4"),
      duration: duration
    )
  }
}