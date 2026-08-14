import Foundation
import Testing

@testable import H3ddleGeneration

@Suite("Generation studio settings")
struct GenerationStudioSettingsTests {
  @Test("Named presets write the documented knob combinations")
  func namedPresets() {
    var settings = GenerationStudioSettings.makeDefault(seed: 1)
    settings.apply(preset: .standard)
    #expect(settings.preset == .standard)
    #expect(settings.knobs.canvas == .square512)
    #expect(settings.knobs.denoisingSteps == 20)
    #expect(settings.knobs.activeDiTLayers == 45)
  }

  @Test("Editing a knob selects Custom and remembers the snapshot")
  func editingSelectsCustom() {
    var settings = GenerationStudioSettings.makeDefault(seed: 1)
    settings.apply(preset: .preview)
    settings.updateKnobs { $0.denoisingSteps = 8 }
    #expect(settings.preset == .custom)
    #expect(settings.custom.denoisingSteps == 8)

    settings.apply(preset: .high)
    #expect(settings.knobs.canvas == .native768)
    #expect(settings.knobs.denoisingSteps == 20)

    settings.apply(preset: .custom)
    #expect(settings.knobs.denoisingSteps == 8)
    #expect(settings.knobs.canvas == .square256)
  }

  @Test("Matching a named preset snaps back to that preset")
  func matchingNamedPreset() {
    var settings = GenerationStudioSettings.makeDefault(seed: 1)
    settings.updateKnobs { $0.denoisingSteps = 9 }
    settings.updateKnobs { $0.denoisingSteps = 4 }
    #expect(settings.preset == .preview)
  }

  @Test("Native landscape flips for portrait")
  func nativeLandscapeFollowsOrientation() {
    #expect(GenerationCanvas.native1344.dimensions(isPortrait: false) == (1344, 768))
    #expect(GenerationCanvas.native1344.dimensions(isPortrait: true) == (768, 1344))
  }
}
