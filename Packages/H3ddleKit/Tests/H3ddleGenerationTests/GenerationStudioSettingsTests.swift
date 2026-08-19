import Foundation
import Testing

import H3ddleEngineProtocol
@testable import H3ddleGeneration

@Suite("Generation studio settings")
struct GenerationStudioSettingsTests {
  @Test("Only video engines with a still path appear as image models")
  func videoEngineStillSupport() {
    #expect(VideoGenerationEngine.h3.supportsStillGeneration)
    #expect(!VideoGenerationEngine.ltx.supportsStillGeneration)
  }

  @Test("Named presets write the documented knob combinations")
  func namedPresets() {
    var settings = GenerationStudioSettings.makeDefault(seed: 1)
    settings.apply(preset: .standard)
    #expect(settings.preset == .standard)
    #expect(settings.knobs.canvas == .p512)
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
    #expect(settings.knobs.canvas == .p768)
    #expect(settings.knobs.denoisingSteps == 20)

    settings.apply(preset: .custom)
    #expect(settings.knobs.denoisingSteps == 8)
    #expect(settings.knobs.canvas == .p256)
  }

  @Test("Matching a named preset snaps back to that preset")
  func matchingNamedPreset() {
    var settings = GenerationStudioSettings.makeDefault(seed: 1)
    settings.updateKnobs { $0.denoisingSteps = 9 }
    settings.updateKnobs { $0.denoisingSteps = 4 }
    #expect(settings.preset == .preview)
  }

  @Test("Landscape flips for portrait")
  func nativeLandscapeFollowsOrientation() {
    #expect(H3Canvas.dimensions(aspect: 16.0 / 9) == (1344, 768))
    #expect(H3Canvas.dimensions(aspect: 9.0 / 16) == (768, 1344))
  }
}

@Suite("Canvas ladder")
struct GenerationCanvasTests {
  @Test("The picker contains every validated native H3 tier")
  func validatedNativeTiers() {
    #expect(GenerationCanvas.allCases == [.p256, .p512, .p768])
    #expect(GenerationCanvas.p256.frame(aspect: 16.0 / 9) == (448, 256))
    #expect(GenerationCanvas.p512.frame(aspect: 16.0 / 9) == (896, 512))
    #expect(GenerationCanvas.p768.frame(aspect: 16.0 / 9) == (1344, 768))
  }

  @Test("Every canvas the app can ask for is inside H3's envelope")
  func everyCanvasIsInsideTheEnvelope() {
    for canvas in GenerationCanvas.allCases {
      for aspect in [16.0 / 9, 1.0, 9.0 / 16, 4.0 / 5, 3.0 / 2, 2.39, 0.4] {
        let size = canvas.frame(aspect: aspect)
        #expect(min(size.width, size.height) <= canvas.shortEdge)
        #expect(size.width * size.height <= H3Canvas.maximumPixels)
        #expect(size.width % 32 == 0)
        #expect(size.height % 32 == 0)
      }
    }
  }

  @Test("Resolution tiers retain their matching work presets")
  func tiersRetainTheirWorkPresets() {
    for canvas in GenerationCanvas.allCases {
      switch canvas {
      case .p256: #expect(canvas.engineQuality == .preview)
      case .p512: #expect(canvas.engineQuality == .standard)
      case .p768: #expect(canvas.engineQuality == .high)
      }
    }
  }

  @Test("Settings saved under either old ladder still load")
  func legacyDecoding() throws {
    let decoder = JSONDecoder()
    #expect(try decoder.decode(GenerationCanvas.self, from: Data("\"square256\"".utf8)) == .p256)
    #expect(try decoder.decode(GenerationCanvas.self, from: Data("\"p352\"".utf8)) == .p256)
    #expect(try decoder.decode(GenerationCanvas.self, from: Data("\"p480\"".utf8)) == .p512)
    #expect(try decoder.decode(GenerationCanvas.self, from: Data("\"p576\"".utf8)) == .p512)
    #expect(try decoder.decode(GenerationCanvas.self, from: Data("\"native1344\"".utf8)) == .p768)
    #expect(try decoder.decode(GenerationCanvas.self, from: Data("\"p1088\"".utf8)) == .p768)
    #expect(try decoder.decode(GenerationCanvas.self, from: Data("\"p768\"".utf8)) == .p768)
  }

  @Test("A degenerate aspect falls back to square rather than crashing")
  func degenerateAspect() {
    #expect(H3Canvas.dimensions(aspect: 0) == (768, 768))
    #expect(H3Canvas.dimensions(aspect: .nan) == (768, 768))
  }

  /// Transcribed from the released pipeline. These are the numbers the model
  /// was trained on, and the app was outside both of them.
  @Test("H3's canvas matches the reference's adapt_canvas")
  func h3CanvasMatchesReference() {
    // 16:9 nominal is 1365x768, over the area cap, and scales to 1344x768.
    #expect(H3Canvas.dimensions(aspect: 16.0 / 9) == (1344, 768))
    #expect(H3Canvas.dimensions(aspect: 9.0 / 16) == (768, 1344))
    #expect(H3Canvas.dimensions(aspect: 1.0) == (768, 768))
    // The short edge never depends on what the caller asked for.
    for aspect in [16.0 / 9, 4.0 / 5, 1.0, 9.0 / 16, 3.0 / 2] {
      let size = H3Canvas.dimensions(aspect: aspect)
      #expect(min(size.width, size.height) == H3Canvas.shortEdge)
      #expect(size.width * size.height <= H3Canvas.maximumPixels)
      #expect(size.width % 32 == 0 && size.height % 32 == 0)
    }
  }

  @Test("Durations land on the 17k+5 grid inside the trained range")
  func h3DurationStaysInRange() {
    #expect(H3Duration.aligned(frames: 5) == 124)
    #expect(H3Duration.aligned(frames: 73) == 124, "73 was below the trained range")
    #expect(H3Duration.aligned(frames: 124) == 124)
    #expect(H3Duration.aligned(frames: 125) == 141)
    #expect(H3Duration.aligned(frames: 9999) == H3Duration.maximumFrames)
    for frames in stride(from: 124, through: 362, by: 17) {
      #expect(H3Duration.aligned(frames: frames) == frames)
      #expect((frames - 5) % 17 == 0)
    }
  }

  /// The snapshot writes its own coding keys, so a new knob has to be added
  /// in four places and reaches the wire only if every one of them agrees.
  /// A round trip is the cheapest way to catch the one that was missed.
  @Test("The image canvas survives a round trip and defaults when absent")
  func imageCanvasCoding() throws {
    let knobs = GenerationKnobSnapshot(
      canvas: .p768,
      imageCanvas: .p1280,
      denoisingSteps: 8,
      activeDiTLayers: 30,
      coreReuse: 1
    )
    let encoded = try JSONEncoder().encode(knobs)
    #expect(try JSONDecoder().decode(GenerationKnobSnapshot.self, from: encoded) == knobs)

    // Settings written before the knob existed carry no key at all.
    let legacy = Data(
      #"{"canvas":"p768","denoisingSteps":8,"activeDiTLayers":30,"coreReuse":1}"#.utf8
    )
    let restored = try JSONDecoder().decode(GenerationKnobSnapshot.self, from: legacy)
    #expect(restored.imageCanvas == .p1024)
  }

  /// Every tier the picker offers has to be one the renderer accepts: a
  /// multiple of 16 whose token count is a multiple of 32. 1440 looks like it
  /// belongs in this list and does not qualify, which is why it is asserted
  /// rather than eyeballed.
  @Test("Every offered image canvas is one the renderer can actually draw")
  func imageCanvasesAreRenderable() {
    for canvas in ImageCanvas.allCases {
      #expect(canvas.label == "\(canvas.shortEdge)p")
      // Every tier at every aspect a project can be in, because the frame is
      // no longer a square and the token rule bites on the product of the two
      // sides rather than on one of them.
      for aspect in [16.0 / 9, 9.0 / 16, 1.0, 4.0 / 5, 3.0 / 2] {
        let frame = canvas.frame(aspect: aspect)
        #expect(frame.width % 16 == 0, "\(frame.width) is not a multiple of 16")
        #expect(frame.height % 16 == 0, "\(frame.height) is not a multiple of 16")
        let tokens = (frame.width / 16) * (frame.height / 16)
        #expect(tokens % 32 == 0, "the token count must divide by 32")
        // The tier's short edge survives the stretch; only the long one moves.
        #expect(min(frame.width, frame.height) == canvas.shortEdge)
      }
    }
  }
}
