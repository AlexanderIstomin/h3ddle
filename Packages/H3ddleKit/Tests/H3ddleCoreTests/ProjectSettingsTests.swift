import Foundation
import Testing

@testable import H3ddleCore

@Suite("Project settings")
struct ProjectSettingsTests {
  @Test("Aspect keeps the long edge and reflows the other")
  func aspectKeepsLongEdge() {
    var settings = ProjectSettings(width: 1920, height: 1080)
    settings.apply(aspect: .nineSixteen)
    #expect(settings.width == 1080)
    #expect(settings.height == 1920)
    #expect(settings.aspect == .nineSixteen)
    #expect(settings.platform == .custom)
  }

  @Test("Platform presets set size and frame rate")
  func platformPreset() {
    var settings = ProjectSettings()
    settings.apply(platform: .instagramFeed)
    #expect(settings.width == 1080)
    #expect(settings.height == 1350)
    #expect(settings.framesPerSecond == 30)
  }

  @Test("Resolution keeps the current aspect and scales the short edge")
  func resolutionKeepsAspect() {
    var settings = ProjectSettings(width: 1080, height: 1920)
    settings.apply(resolution: .fullHD)
    #expect(settings.width == 1080)
    #expect(settings.height == 1920)
    #expect(settings.resolution == .fullHD)
    #expect(settings.platform == .custom)

    settings.apply(resolution: .ultraHD)
    #expect(settings.width == 2160)
    #expect(settings.height == 3840)

    settings.apply(aspect: .fourFive)
    settings.apply(resolution: .fullHD)
    #expect(settings.width == 1080)
    #expect(settings.height == 1350)
  }

  @Test("Master gain and exposure stay in range")
  func clampsOutputControls() {
    let settings = ProjectSettings(masterGain: 4, exposure: 0)
    #expect(settings.masterGain == 1)
    #expect(settings.exposureStops == -3)

    var next = ProjectSettings()
    next.exposure = pow(2, 1.5)
    #expect(abs(next.exposureStops - 1.5) < 0.000_1)
  }

  @Test("Legacy projects decode without settings")
  func decodesLegacyProjects() throws {
    let project = H3ddleProject(name: "Legacy")
    var encoded = try JSONEncoder().encode(project)
    var object = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
    object.removeValue(forKey: "settings")
    encoded = try JSONSerialization.data(withJSONObject: object)
    let decoded = try JSONDecoder().decode(H3ddleProject.self, from: encoded)
    #expect(decoded.settings.width == 1920)
    #expect(decoded.settings.framesPerSecond == 24)
    #expect(decoded.settings.toneMapping == .none)
    #expect(decoded.settings.masterGain == 1)
  }
}
